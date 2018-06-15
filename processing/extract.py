import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import argparse
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import SimpleITK as sitk
import csv
import cv2
import math

def worldToVox(centroids_world,spacing,scanOrigin):
    '''
    convert centroids from world frame (extracted from csv files, into voxel frame
    Inputs: 
        centroids_world : list of centroids in world coordinates 
        spacing: list of voxel spacings in mm 
        scanOrigin: location of voxel frame origin wrt world frame
    Outputs:
        centroids_vox: list of centroids in voxel coordinates
    note: the following only works if the voxel frame is not rotated with respect to the world frame - which is the case
    for all the data used
    '''
    centroids_world = np.array(centroids_world).astype(float)
    origin = np.array(scanOrigin).astype(float)
    scaled_vox = np.absolute(centroids_world - origin) #broadcast
    centroids_vox = scaled_vox / spacing.reshape((1,3))
    centroids_vox = centroids_vox.astype(int)
    return centroids_vox.tolist()	

def getCandidates(is_pos,file_path):
    '''
    extract candidate locations from CSV file
    Output: dictionary with series UID of a scan as keys and a list of nodule centroids as values
    '''
    candidates = {}
    count = 0 
    max_count = math.inf
    if not is_pos:
        max_count = 400    
    with open(file_path, newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in annotations:
            series_uid = row[0]
            x = row[1]
            y = row[2]
            z = row[3]
            centroid = [z,y,x]
            if series_uid in candidates:
                if len(candidates[series_uid]) < max_count:
                    candidates[series_uid].append(np.array(centroid))
                    count = count+1
            else:
                candidates[series_uid] = [np.array(centroid)]
                count = count+1
    print(str(count) + " relevant nodules in csv file")
    return nodules

def load_itk_image(filename):
    '''
    load scan volume from mhd file
    inputs: path to file
    outputs: scan volume, scan origin with respect to world frame, voxel spacings in mm
    '''
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def loadScan(path): 
    #source:GuidoZuidhof link:https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook
    '''
    load scan volume from dicom files
    inputs: path to dicom file
    outputs: list of slices in a scan volume
    '''
    
    slices = [pydicom.read_file(os.path.join(path,s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) 
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def getPixels(slices,image,dicom=0):
    '''
    convert voxel values to Housenfeild units
    normalize and threshold 
    inputs: slices (if dicom is used), image (if mhd files are used), dicom=1 if dicom is used
    outputs: normalized volume
    '''
    if dicom==1:
        raw_image = np.stack([s.pixel_array for s in slices])
        raw_image = raw_image.astype(np.int16)
     
    # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
        
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
        
            if slope != 1:
                raw_image[slice_number] = slope * raw_image[slice_number].astype(np.float64)
                raw_image[slice_number] = raw_image[slice_number].astype(np.int16)
            
            raw_image[slice_number] += np.int16(intercept)
    
        raw_image = np.array(raw_image)
        image=raw_image
    

    maxHU = 400
    minHU = -1000
    image = (image - minHU)/(maxHU - minHU)
    image[image>1] = 1
    image[image<0] = 0
    return image

def extractCandidate(volume,cen_vox,vox_size,spacing,new_spacing,translate): 
    '''
    given a centroid location in the volume, calculate the sub volume shape (old_shape) that would result
    in the desired sub volume shape after resampling at a new spacing
    Inputs: 
        volume: scan volume
        cen_vox: centroid in voxel coordinates
        spacing: list of original scan voxel spacings
        new_spacing: list of desired voxel spacings
        translations: list of translations for augmentation
    Outputs: 
        sub_vols: list of extracted sub volumes, 
        patches : 2d plane centered at centroid, used for visualization
    '''
    sub_vols = []
    patches = []
    resize_factor = spacing/new_spacing
    old_shape = np.round(vox_size/resize_factor)
    resize_factor = vox_size/old_shape
    new_spacing = spacing/resize_factor
    top_left = (cen_vox - old_shape/2).astype(int) 
    bot_right = (cen_vox + old_shape/2).astype(int) 
    sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
    sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
    assert np.shape(sub_volume) == (32,32,32) # using this as a lazy way to reject extracted sub volumes that dont fall within the original volume's bounds
    patch = sub_volume[int(vox_size[0]/2),:,:]
    sub_vols.append(sub_volume)
    patches.append(patch)
    if translate!=0:
        translations = np.random.randint(-SUBVOL_DIM[0]/4,high=SUBVOL_DIM[0]/4,size=(translate,1,3)) 
        for translation in translations:
            trans_np = np.squeeze(translation)
            top_left = (cen_vox - old_shape/2).astype(int) + trans_np
            bot_right = (cen_vox + old_shape/2).astype(int) + trans_np
            sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
            sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
            assert np.shape(sub_volume) == (32,32,32) # using this as a lazy way to reject extracted sub volumes that dont fall within the original volume's bounds
            patch = sub_volume[int(vox_size[0]/2-trans_np[0]),:,:]
            sub_vols.append(sub_volume)
            patches.append(patch)
    return sub_vols, patches




