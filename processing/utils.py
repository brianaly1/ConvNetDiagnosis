import os
import sys
import settings
import glob
import math
import numpy as np
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
    vol = np.vstack(images)
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
        res = np.vstack([res1, res2])
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

def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res

def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)

def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    res = np.zeros((rows * cols, size, size))

    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res

def print_tabbed(value_list, justifications=None, map_id=None, show_map_idx=True):
    map_entries = None
    if map_id is not None:
        map_entries = PRINT_TAB_MAP[map_id]

    if map_entries is not None and show_map_idx:
        idx = str(len(map_entries))
        if idx == "0":
            idx = "idx"
        value_list.insert(0, idx)
        if justifications is not None:
            justifications.insert(0, 6)

    value_list = [str(v) for v in value_list]
    if justifications is not None:
        new_list = []
        assert(len(value_list) == len(justifications))
        for idx, value in enumerate(value_list):
            str_value = str(value)
            just = justifications[idx]
            if just > 0:
                new_value = str_value.ljust(just)
            else:
                new_value = str_value.rjust(just)
            new_list.append(new_value)

        value_list = new_list

    line = "\t".join(value_list)
    if map_entries is not None:
        map_entries.append(line)
    print(line)

def prepare_example(volume,vox_size,translations, label): 
    '''
    Extract example from larger volume, optionally apply random translations to examples 
    for augmentation
    Inputs: 
        volume: list of slices, making up volume (np arrays)
        vox_size: desired sub volume size (tuple)
        translations: desired number of translations  
        label: nodule or not 
    Outputs:
        sub_volumes: list of sub volumes (3d np arrays)
        labels: list of labels 
    '''
    
    sub_vols = []

    volume_np = np.array(volume)
    vox_size_np = np.array(vox_size)
    translations = int(translations)
    label = int(label)

    # extract sub volume
    vol_size = np.array(np.shape(volume_np)) 
    margin = (vol_size - vox_size_np)/2
    top_left = margin.astype(int) 
    bot_right = (vol_size - margin).astype(int)  
    sub_volume = volume_np[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]

    assert np.shape(sub_volume) == vox_size, "extracted sub volume shape does not match desired shape"

    sub_vols.append(sub_volume)
    
    #augment with random translations 
    if translations!=1:
        max_shift = min(margin)
        deltas = np.random.randint(low=-max_shift,high=max_shift,size=(translations-1,1,3)) 
        for translation in deltas:
            trans_np = np.squeeze(translation)
            top_left_new = (top_left + trans_np).astype(int)
            bot_right_new = (bot_right + trans_np).astype(int)

            sub_volume = volume[top_left_new[0]:bot_right_new[0],top_left_new[1]:bot_right_new[1],top_left_new[2]:bot_right_new[2]]

            assert np.shape(sub_volume) == vox_size, "extracted sub volume shape does not match desired shape"

            sub_vols.append(sub_volume)

    labels = [label]*len(sub_vols)

    return sub_vols,labels

