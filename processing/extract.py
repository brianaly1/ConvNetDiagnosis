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


NOD_FILE = "/home/alyb/data/CSVFILES/annotations.csv" #csv file with centroids of nodules
FALSE_FILE = "/home/alyb/data/CSVFILES/candidates_V2.csv" #csv file with centroids of false positive

def getNodules():
    '''
    extract nodule locations from CSV file
    Output: dictionary with series UID of a scan as keys and a list of nodule centroids as values
    '''
    nodules = {}
    count = 0
    with open(NOD_FILE, newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in annotations:
            series_uid = row[0]
            x = row[1]
            y = row[2]
            z = row[3]
            centroid = [z,y,x]
            if series_uid in nodules:
                nodules[series_uid].append(np.array(centroid))
                count = count+1
            else:
                nodules[series_uid] = [np.array(centroid)]
                count = count+1
    print(str(count) + " relevant nodules in csv file")
    return nodules

def getCandidates():
    '''
    extract false positive locations from CSV file
    Output: dictionary with series UID of a scan as keys and a list of nodule centroids as values
    '''
    candidates = {}
    count = 0
    with open(FALSE_FILE, newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in annotations:
            series_uid = row[0]
            x = row[1]
            y = row[2]
            z = row[3]
            label = row[4]
            centroid = [z,y,x]
            if int(label)==0:
                if series_uid in candidates: 
                    if len(candidates[series_uid]) < 400:
                        candidates[series_uid].append(np.array(centroid))
                        count = count + 1
                else:
                    candidates[series_uid] = [np.array(centroid)]
                    count = count + 1
    print(str(count) + " relevant candidates in csv file")
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
    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) 
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def getPixels(slices,image,dicom=1):
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
    
        raw_image = np.array(image)

        maxHU = 400
        minHU = -1000
        raw_image = (raw_image - minHU)/(maxHU - minHU)
        raw_image[image>1] = 1
        raw_image[image<0] = 0
        return raw_image
    else:
        maxHU = 400
        minHU = -1000
        image = (image - minHU)/(maxHU - minHU)
        image[image>1] = 1
        image[image<0] = 0
        return image

def resample(image, spacing, new_spacing):
    '''
    resample an input volume at a different voxel spacing by interpolation
    inputs: image (scan volume), spacing (old spacing), new_spacing
    outputs: resampled image
    '''
    spacing = [spacing[2],spacing[1],spacing[0]]
    new_spacing = [new_spacing[2],new_spacing[1],new_spacing[0]]
    # Resizing dim z
    resize_x = 1.0
    resize_y = float(spacing[2]) / float(new_spacing[2])
    interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes [y,x,channels]

    resized = resized.swapaxes(0, 2)
    resized = resized.swapaxes(0, 1)
    resize_x = float(spacing[1]) / float(new_spacing[1])
    resize_y = float(spacing[0]) / float(new_spacing[0])

    # cv2 can handle max 512 channels
    if resized.shape[2] > 512:
        resized = resized.swapaxes(0, 2)
        res1 = np.array(resized[:256])
        res2 = np.array(resized[256:])
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        resized = np.vstack([res1, res2])
        resized = resized.swapaxes(0, 2)
    else:
        resized = cv2.resize(resized, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    resized = resized.swapaxes(0, 2)
    resized = resized.swapaxes(2, 1)

    return resized

def resize(image,new_shape):

    resize_x = 1.0
    interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, dsize=(new_shape[1], new_shape[0]), interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    resized = res.swapaxes(0, 2)
    resized = res.swapaxes(0, 1)

    # cv2 can handle max 512 channels..
    if resized.shape[2] > 512:
        resized = resized.swapaxes(0, 2)
        res1 = resized[:256]
        res2 = resized[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        resized = np.vstack([res1, res2])
        resized = resized.swapaxes(0, 2)
    else:
        resized = cv2.resize(resized, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)

    resized = resized.swapaxes(0, 2)
    resized = resized.swapaxes(2, 1)

    return resized


