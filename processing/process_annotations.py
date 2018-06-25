import os
import sys
import SimpleITK
import utils
import settings
import pandas
import numpy as np
import math
import glob
import cv2
import random


def process_pos_annotation(path, uid):

    annotations_path = os.path.join(settings.PATIENTS_DIR,"annotations","annotations.csv") 
    df_node = pandas.read_csv(annotations_path)

    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == uid]
    print("Annos: ", len(df_patient))
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing /settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)
    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swapping x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swapping y origin")
    print("Direction: ", direction)

    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = utils.load_patient_images(uid, "_i.png")

    pos_annos = []
    anno_index = 0

    for index, annotation in df_patient.iterrows():

        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        diam_mm = annotation["diameter_mm"]

        print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(diam_mm, 2)))

        center_float = np.array([node_x, node_y, node_z])
        center_int = np.rint((center_float-origin) / spacing)
        print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])
        pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_annos_pos.csv")
    df_annos.to_csv(save_path, index=False)

    return [uid, spacing[0], spacing[1], spacing[2]]


def process_excluded_annotation(path, uid):
    annotations_path = os.path.join(settings.PATIENTS_DIR,"annotations","annotations_excluded.csv") 
    df_node = pandas.read_csv(annotations_path)

    annotations_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_annos_pos.csv")
    pos_annos_df = pandas.read_csv(annotations_path)
    

    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == uid]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swapping x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swapping y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = utils.load_patient_images(uid, "_i.png")

    neg_annos = []
    df_patient = df_node[df_node["seriesuid"] == uid]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        center_float = np.array([node_x, node_y, node_z])
        center_int = np.rint((center_float-origin) / spacing)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        diameter_pixels = 6 / settings.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        ok = True

        for index, row in pos_annos_df.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            print((pos_coord_x, pos_coord_y, pos_coord_z))
            print(center_float_rescaled)
            dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
            if dist < (diameter + 64):  #  make sure we have a big margin
                ok = False
                print("################### Too close", center_float_rescaled)
                break

        if not ok:
            continue

        neg_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1
    
    df_annos = pandas.DataFrame(neg_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_annos_excluded.csv")
    df_annos.to_csv(save_path, index=False)
    return [uid, spacing[0], spacing[1], spacing[2]]

def process_luna_candidates(path, uid):

    annotations_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_annos_pos_lidc.csv") 
    df_pos_annos = pandas.read_csv(annotations_path)

    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swapping x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swapping y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    annotations_path = os.path.join(settings.PATIENTS_DIR,"annotations","candidates_V2.csv")
    src_df = pandas.read_csv(annotations_path)
    src_df = src_df[src_df["seriesuid"] == uid]
    src_df = src_df[src_df["class"] == 0]

    patient_imgs = utils.load_patient_images(uid, "_i.png")

    candidate_list = []

    for df_index, candiate_row in src_df.iterrows():
        node_x = candiate_row["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = candiate_row["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = candiate_row["coordZ"]
        candidate_diameter = 6
        center_float = np.array([node_x, node_y, node_z])
        center_int = np.rint((center_float-origin) / spacing)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        coord_x = center_float_rescaled[0]
        coord_y = center_float_rescaled[1]
        coord_z = center_float_rescaled[2]

        ok = True

        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 64):  #  make sure we have a big margin
                ok = False
                print("################### Too close", (coord_x, coord_y, coord_z))
                break

        if not ok:
            continue

        candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_candidates_luna.csv")
    df_candidates.to_csv(save_path, index=False)

def process_auto_candidates(path, uid, sample_count=1000, candidate_type="white"):
    annotations_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_annos_pos_lidc.csv") 
    df_pos_annos = pandas.read_csv(annotations_path)
    img_dir = os.path.join(settings.LUNA_IMAGE_DIR,uid) + "/"


    itk_img = SimpleITK.ReadImage(path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    if candidate_type == "white":
        wildcard = "*_c.png"
    else:
        wildcard = "*_m.png"

    src_files = glob.glob(img_dir + wildcard)
    src_files.sort()
    src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]

    candidate_list = []
    tries = 0
    while len(candidate_list) < sample_count and tries < 10000:
        tries += 1
        coord_z = int(np.random.normal(len(src_files) / 2, len(src_files) / 6))
        coord_z = max(coord_z, 0)
        coord_z = min(coord_z, len(src_files) - 1)
        candidate_map = src_candidate_maps[coord_z]
        if candidate_type == "edge":
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

        non_zero_indices = np.nonzero(candidate_map)
        if len(non_zero_indices[0]) == 0:
            continue
        nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
        coord_y = non_zero_indices[0][nonzero_index]
        coord_x = non_zero_indices[1][nonzero_index]
        ok = True
        candidate_diameter = 6
        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * src_candidate_maps[0].shape[1]
            pos_coord_y = row["coord_y"] * src_candidate_maps[0].shape[0]
            pos_coord_z = row["coord_z"] * len(src_files)
            diameter = row["diameter"] * src_candidate_maps[0].shape[1]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 48): #  make sure we have a big margin
                ok = False
                print("# Too close", (coord_x, coord_y, coord_z))
                break

        if not ok:
            continue


        perc_x = round(coord_x / src_candidate_maps[coord_z].shape[1], 4)
        perc_y = round(coord_y / src_candidate_maps[coord_z].shape[0], 4)
        perc_z = round(coord_z / len(src_files), 4)
        candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(candidate_diameter / src_candidate_maps[coord_z].shape[1], 4), 0])

    if tries > 9999:
        print("****** WARING!! TOO MANY TRIES ************************************")
    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",uid+"_candidates_"+candidate_type + ".csv")
    df_candidates.to_csv(save_path, index=False)


def process_annotations(mode):
    candidate_index = 0
    patients = utils.load_patients_list()
    for patient in patients:
        print(candidate_index, ":patient: ", patient)
        if mode==0:
            process_pos_annotation(patients[patient], patient)
        elif mode==1:
            process_excluded_annotation(patients[patient], patient)
        elif mode==2:
            process_luna_candidates(patients[patient], patient)   
        elif mode==3:
            process_auto_candidates(patients[patient], patient,sample_count=200, candidate_type="edge")  
        candidate_index += 1
        print('-------------------------------------------')



def main():
    process_annotations(2)
    process_annotations(3)
    
if __name__=="__main__":
    main()
