import os
import settings
import utils
import glob
import pandas
import numpy as np
import cv2
import sys

def make_annotation_images(mode):
    '''
    This function processes sub volumes from the volume images,using the 
    prepared csv files containing centroid locations
    to their respective paths.
    Inputs:
        mode: 0 processes LIDC positives, 1 processes LUNA positives
    '''

    src_dir = os.path.join(settings.CA_PATIENTS_DIR,"labels")

    if mode==0:
        ext = "*_annos_pos_lidc.csv"
        dst_dir = os.path.join(settings.CA_TRAIN_DATA_DIR,"LIDC")
    elif mode==1:
        ext = "*_annos_pos.csv"
        dst_dir = os.path.join(settings.CA_TRAIN_DATA_DIR,"POS")
      
    #patients = utils.load_patients_list()
    patients = os.listdir(settings.CA_LUNA_IMAGE_DIR)

    for patient_index,patient in enumerate(patients):
        csv_dir = os.path.join(src_dir,patient) + '/'
        csv_file = glob.glob(csv_dir + ext)
        df_annos = pandas.read_csv(csv_file[0])

        if len(df_annos) == 0:
            continue

        images = utils.load_patient_images(patient, "_i.png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            diam = int(row["diameter"] * images.shape[2])
            malscore = int(row["malscore"])
            anno_index = row["anno_index"]
            cube_img = utils.get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            if mode==0:
                target_path = os.path.join(dst_dir,patient + "_" + str(anno_index) + "_" + str(malscore * malscore) + "_1_pos.png")
                if malscore==4 or malscore==5:
                    print("not one or two or three")
                    continue

            elif mode==1:
                target_path = os.path.join(dst_dir,patient + "_" + str(anno_index) + "_" + str(diam) + "_1_pos.png")

            utils.save_cube_img(target_path, cube_img, 8, 8)

        utils.print_tabbed([patient_index, patient, len(df_annos)], [5, 64, 8])

def make_candidate_images(mode):
    '''
    This function processes sub volumes from the volume images,using the 
    prepared csv files containing centroid locations
    to their respective paths.
    Inputs:
        mode: 0 processes LUNA candidates, 1 processes EDGE random candidates 
                                                    
    '''
    src_dir = os.path.join(settings.CA_PATIENTS_DIR,"labels")

    if mode==0:
        ext = "*_candidates_luna.csv"
        dst_dir = os.path.join(settings.CA_TRAIN_DATA_DIR,"NEG")
    elif mode==1:
        ext = "*_candidates_edge.csv"
        dst_dir = os.path.join(settings.CA_TRAIN_DATA_DIR,"EDGE")
   
    patients = utils.load_patients_list()

    for patient_index,patient in enumerate(patients):
        csv_dir = os.path.join(src_dir,patient) + '/'
        csv_file = glob.glob(csv_dir + ext)
        df_annos = pandas.read_csv(csv_file[0])

        if len(df_annos) == 0:
                continue

        images = utils.load_patient_images(patient, "_i.png")

        row_no = 0

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            anno_index = int(row["anno_index"])
            cube_img = utils.get_cube_from_img(images, coord_x, coord_y, coord_z, 64)

            if cube_img.sum() < 10:
                print("Skipping ", coord_x, coord_y, coord_z)
                continue

            if mode==0:
                target_path = os.path.join(dst_dir,patient + "_" + str(anno_index) + "_0_" + "luna.png")

            elif mode==1:
                target_path = os.path.join(dst_dir,patient + "_" + str(anno_index) + "_0_" + "edge.png")

            try:
                utils.save_cube_img(target_path, cube_img, 8, 8)

            except Exception as ex:
                print(ex)

            row_no += 1
            max_item = 500 if mode==0 else 200
            if row_no > max_item:
                break
        
        utils.print_tabbed([patient_index, patient, len(df_annos)], [5, 64, 8])

def main():
    make_annotation_images(0)
    #make_annotation_images(1)
    #make_candidate_images(0)
    #make_candidate_images(1)
    
if __name__ == "__main__":
    main()
            

