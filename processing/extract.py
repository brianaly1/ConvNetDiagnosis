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
    return candidates

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
        volume: scan volume - np array
        cen_vox: centroid in voxel coordinates - np array
        spacing: original scan voxel spacings - list
        new_spacing: desired voxel spacings - list
        translations: translations for augmentation - list
    Outputs: 
        sub_vols: extracted sub volumes - list of np arrays 
        patches : 2d plane centered at centroid, used for visualization - np array
    '''
    
    sub_vols = []
    patches = []
    
    spacing = np.array(spacing)
    new_spacing = np.array(new_spacing)
    # extract sub volume with old_shape such that after new_spacing is applies, vox_size is obtained
    resize_factor = spacing/new_spacing
    old_shape = np.round(vox_size/resize_factor)
    resize_factor = vox_size/old_shape
    new_spacing = spacing/resize_factor
    
    # extract sub volume
    top_left = (cen_vox - old_shape/2).astype(int) 
    bot_right = (cen_vox + old_shape/2).astype(int) 
    for i in range(3):
        delta = (bot_right[i] - top_left[i]) - old_shape[i]
        if delta<0:
            bot_right[i] -= delta
        elif delta>0:
            bot_right[i] -= delta
    sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
    if np.any(resize_factor != 1.0) == True:
        sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
    assert np.shape(sub_volume) == (32,32,32) # to reject extracted sub volumes that dont fall within the original volume's bounds

    patch = sub_volume[int(vox_size[0]/2),:,:]

    sub_vols.append(sub_volume)
    patches.append(patch)
    
    #augment with random translations 
    if translate!=0:
        translations = np.random.randint(-SUBVOL_DIM[0]/4,high=SUBVOL_DIM[0]/4,size=(translate,1,3)) 
        for translation in translations:
            trans_np = np.squeeze(translation)
            top_left = ((cen_vox - old_shape/2) + trans_np).astype(int)
            bot_right = ((cen_vox + old_shape/2) + trans_np).astype(int)

            sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
            sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
            assert np.shape(sub_volume) == (32,32,32) 

            patch = sub_volume[int(vox_size[0]/2-trans_np[0]),:,:]

            sub_vols.append(sub_volume)
            patches.append(patch)

    return sub_vols, patches

def segmentVolume(vol_shape, centroids, sub_vol_shape, spacing, new_spacing):
    '''
    Segment a volume into sub volumes, and generate labels for each
    Inputs:
        vol_shape: size of the volume - np array
        centroids: locations of nodules in the volume - list
        sub_vol_shape: desired sub volume shape - np array
        spacing: current voxel spacing - list
        new_spacing: desired voxel spacing - list
    Outputs:
        sub_vol_centroids: centroids of the individual sub volumes
        labels: label of each centroid
    '''

    sub_vol_centroids = []
    labels = []

    resize_factor = spacing/new_spacing
    old_shape = np.round(sub_vol_shape/resize_factor)
    k_step = int(old_shape[0])
    i_step = int(old_shape[1])
    j_step = int(old_shape[2])

    limits = (vol_shape-old_shape/2).astype(int)
    for k in range(k_step//2,limits[0],k_step):
        for i in range(i_step//2,limits[1],i_step):
            for j in range(j_step//2,limits[2],j_step):
                centroid = [k,i,j]
                centroid_np = np.array(centroid)
                label = 0
                if centroids != None:
                    for nodule in centroids:
                        nodule_np = np.array(nodule)
                        if np.all(np.absolute(centroid_np-nodule_np)<old_shape) == True:
                            label = 1 
                sub_vol_centroids.append(centroid_np)
                labels.append(label)
    sub_vol_centroids = np.array(sub_vol_centroids)
    return sub_vol_centroids, labels
             
    



