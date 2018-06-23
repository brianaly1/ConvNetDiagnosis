import os
import shutil
import settings
import utils
import cv2
import SimpleITK
import numpy as np


def normalize(image):

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_image(uid,path):

    print("Patient: ", uid)
    dst_dir = os.path.join(settings.LUNA_IMAGE_DIR,uid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)
    img_array = utils.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)
    img_list = []
    for index,img in enumerate(img_array):
        seg_img, mask = utils.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        img_name = "img_%s_%s" % (str(index).rjust(4, '0'),"_i.png")
        mask_name = "img_%s_%s" % (str(index).rjust(4, '0'),"_m.png")
        cv2.imwrite(os.path.join(dst_dir,img_name), img * 255)
        cv2.imwrite(os.path.join(dst_dir,mask_name), mask * 255)
        

def process_images(delete_existing=True):

    if delete_existing and os.path.exists(settings.LUNA_IMAGE_DIR):
        print("Removing old stuff..")
        shutil.rmtree(settings.LUNA_IMAGE_DIR)
    if not os.path.exists(settings.LUNA_IMAGE_DIR):
        os.mkdir(settings.LUNA_IMAGE_DIR)
    patients = utils.load_patients_list()
    for patient in patients:
        process_image(uid=patient,path=patients[patient])
        print('-------------------------------------------')

def main():
    process_images()
    
if __name__=="__main__":
    main()
