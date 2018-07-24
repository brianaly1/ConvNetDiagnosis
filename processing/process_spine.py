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
import xml.etree.ElementTree as ET

def cv_flip(img,cols,rows,degree):
    A = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1.0)
    dst = cv2.warpAffine(img, A, (cols,rows))
    return dst

def parse_xml(xml_file,path):

    xml_path = os.path.join(path,xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    study = root[0].attrib["UID"]
    series = [x for x in root[0] if x.tag=="Series"]
    series_dict = {}

    for element in series: 
        instances = []    
        for inst in element:    
            if inst.tag == "BaseInstance":
                nested_instances = [x for x in inst]
                instances.extend(nested_instances)
            elif inst.tag == "Instance":
                instances.append(inst)
        series_dict[element.attrib["UID"]] = instances
    
    return series_dict

def restructure(path,series_dict):

    for series in series_dict:

        series_dir_dic = os.path.join(path,series)

        if not os.path.exists(series_dir_dic):
            os.mkdir(series_dir_dic)   
        print("num of instances {}".format(len(series_dict[series])))
        for instance in series_dict[series]:
            if instance.attrib["UID"] != "":
                src = os.path.join(path,instance.attrib["UID"]+".dcm")
                dst = os.path.join(series_dir_dic,instance.attrib["UID"]+".dcm")
                if not os.path.exists(dst):
                    shutil.move(src,dst)

    remains = [x for x in os.listdir(path) if x[-4:]==".dcm"]

    if len(remains) != 0:

        if not os.path.isdir(os.path.join(path,"remains")):
            os.mkdir(os.path.join(path,"remains"))

        for dicom_file in remains:
            src = os.path.join(path,dicom_file)
            dst = os.path.join(path,"remains",dicom_file)
            if not os.path.exists(dst):
                shutil.move(src,dst)    

def process_patient(uid,path,series_dict):

    print("Patient: ", uid)

    dst_dir = os.path.join(settings.SP_IMAGE_DIR,uid)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    series_paths = [os.path.join(path,x) for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
    series_uids = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]

    for index,series in enumerate(series_paths):

        try:

            series_dst_dir = os.path.join(dst_dir,series_uids[index])
            if not os.path.exists(series_dst_dir):
                os.mkdir(series_dst_dir)
        
            slices = utils.load_scan(series)
    
            #cos_value = (slices[0].ImageOrientationPatient[0])
            #cos_degree = round(math.degrees(math.acos(cos_value)),2)

            pixels = utils.get_pixels(slices)
            image = pixels

            print("Volume shape is: {}".format(image.shape))

            #invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]

            #pixel_spacing = slices[0].PixelSpacing
            #pixel_spacing.append(slices[0].SliceThickness)
            #pixel_spacing = [float(s) for s in pixel_spacing]
            #print("Pixel spacing is: {}".format(pixel_spacing))
            #image = utils.resample_images(image,pixel_spacing,settings.TARGET_VOXEL_MM)
    
            #if not invert_order:
                #image = np.flipud(image)

            for index,img in enumerate(image):
        
                #if cos_degree>0.0:
                    #img = cv_flip(img,img.shape[1],img.shape[0],cos_degree)

                img = utils.normalize(img)
                img_name = "img_%s_%s" % (str(index).rjust(4, '0'),"_i.png")
                save_path = os.path.join(series_dst_dir,img_name)
                cv2.imwrite(save_path, img * 255)

        except Exception as e:

            print("Series: {}, Exception: {}".format(series,e))

def process_scans():

    patients = os.listdir(settings.SP_SRC_DIR)
    for patient in patients:
        xml_file = [x for x in os.listdir(os.path.join(settings.SP_SRC_DIR,patient)) if x[-4:]==".xml"]
        if not xml_file:
            continue
        xml_file = xml_file[0]
        path = os.path.join(settings.SP_SRC_DIR,patient)
        series_dict = parse_xml(xml_file,path)
        restructure(path,series_dict)
        process_patient(uid=patient,path=path,series_dict=series_dict)
        print('-------------------------------------------')


def main():
    process_scans()
    
if __name__=="__main__":
    main()
