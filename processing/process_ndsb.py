import os
import sys
import shutil
import settings
import utils
import cv2
import SimpleITK
import numpy as np
import pydicom
import math

def process_scan(uid,path):
    '''
    This function processes dicom files, rescales the patient volume
    and saves the volume as png images
    Inputs:
        uid: patient uid
        path: path to dicom folder for this patient
    '''

    print("Patient: ", uid)

    dst_dir = os.path.join(settings.CA_NDSB_IMG_DIR,uid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    slices = utils.load_scan(path)
    
    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)

    pixels = utils.get_pixels(slices)
    image = pixels

    print("Volume shape is: {}".format(image.shape))

    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    pixel_spacing = [float(s) for s in pixel_spacing]
    print("Pixel spacing is: {}".format(pixel_spacing))

    image = utils.rescale_patient_images(image,pixel_spacing,settings.TARGET_VOXEL_MM)
    print("Rescaled shape: {}".format(image.shape))


    img_list = []    

    if not invert_order:
        image = np.flipud(image)

    for index2,img in enumerate(image):
        seg_img, mask = utils.get_segmented_lungs(img.copy()) 
        img_list.append(seg_img)
        img = utils.normalize(img)       
        if cos_degree>0.0:
            img = cv_flip(img,img.shape[1],img.shape[0],cos_degree)
        img_name = "img_%s_%s" % (str(index2).rjust(4, '0'),"_i.png")
        mask_name = "img_%s_%s" % (str(index2).rjust(4, '0'),"_m.png")
        cv2.imwrite(os.path.join(dst_dir,img_name), img * 255)
        cv2.imwrite(os.path.join(dst_dir,mask_name), mask * 255)
    

def process_scans(ndsb_dir):
    
    patients = os.listdir(ndsb_dir)
    done_patients = set(os.listdir(ndsb_dir))
    for patient in patients:
        if patient in done_patients:
            try:
                process_scan(uid=patient,path=os.path.join(ndsb_dir,patient))
                print('-------------------------------------------')
            except KeyboardInterrupt:
                print("Program Interrupted")
                sys.exit()
            except Exception as e:
                print("Error: {}".format(e))

def main():
    path = settings.CA_NDSB_SRC_DIR
    process_scans(path)
    
if __name__=="__main__":
    main()
