import os
import sys
import settings
import glob
import math
import numpy
import cv2
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from collections import defaultdict
from scipy import ndimage as ndi
import pandas

def load_patients_list():
    LUNA_DIR = settings.LUNA_SRC_DIR
    subsets = os.listdir(LUNA_DIR)
    patients = {}
    for subset in subsets:
        subset_path = os.path.join(LUNA_DIR,subset)
        subset_files = [x for x in os.listdir(subset_path) if x[-4:] == ".mhd"]   
        subset_series_uids = [x[:-4] for x in subset_files]   
        for patient in subset_files:
            series_uid = patient[:-4]
            path = os.path.join(subset_path,patient)
            patients[series_uid] = path
        
    return patients

def load_patient_images(uid, extension):
    dir_path = os.path.join(settings.LUNA_IMAGE_DIR,uid)
    img_names = os.listdir(dir_path)
    ext_len = int(len(extension) * -1)
    img_paths = [os.path.join(dir_path,img_name) for img_name in img_names if img_name[ext_len:]==extension]
    img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in img_paths]
    images = [img.reshape((1, ) + img.shape) for img in images]
    vol = numpy.vstack(images)
    return vol

def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False):

    # Resizing dim z
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  
    # opencv assumes y, x, channels umpy array, so y = z pfff

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)

    return res


def get_segmented_lungs(im, plot=False):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    regprops = regionprops(label_image)
    areas = [r.area for r in regprops]
    areas.sort()
    if len(areas) > 2:
        for region in regprops:
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return im, binary






