import os
import sys
import shutil
import settings
import utils
import cv2
import SimpleITK
import numpy as np


def normalize(image):
    '''
    Normalization as suggested by the Kaggle NDSB community 
    Inputs:
        image: patient volume - np array
    '''
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_image(uid,path):
    '''
    This function reads a volume (mhd/raw format) and saves the images, along with
    a segmented mask of the lungs, as png.
    Inputs:
        uid: patient uid
        path: path to image
    '''

    #print("Patient: ", uid)
    dst_dir = os.path.join(settings.CA_LUNA_IMAGE_DIR,uid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    #print("Img array: ", img_array.shape)
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    #print("Origin (x,y,z): ", origin)
    direction = np.array(itk_img.GetDirection())  # x,y,z  Origin in world coordinates (mm)
    #print("Direction: ", direction)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    #print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    #print("Rescale: ", rescale)
    img_array = utils.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)
    img_list = []

    for index,img in enumerate(img_array):
        seg_img, mask = utils.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        img_name = "img_%s_%s" % (str(index).rjust(4, '0'),"_i.png")
        mask_name = "img_%s_%s" % (str(index).rjust(4, '0'),"_m.png")
        #print("Test writing to: {}".format(os.path.join(dst_dir,img_name)))
        cv2.imwrite(os.path.join(dst_dir,img_name), img * 255)
        cv2.imwrite(os.path.join(dst_dir,mask_name), mask * 255)
        

def process_images(luna_dir):
 
    patients = utils.load_patients_list()
    done_patients = set(os.listdir(luna_dir))
    for patient in patients:
        if patient not in done_patients:
            try:
                process_image(uid=patient,path=patients[patient])
                print('-------------------------------------------')
            except KeyboardInterrupt:
                print("Program Interrupted")
                break
            except Exception as e:
                print("Error: {}".format(e))
        

def main():
    path = settings.CA_LUNA_SRC_DIR    
    process_images(path)
    
if __name__=="__main__":
    main()
