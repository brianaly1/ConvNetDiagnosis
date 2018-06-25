import os
import sys
import settings
import utils
import math
import numpy
import pandas
import SimpleITK
from bs4 import BeautifulSoup

def load_lidc_xml(patients, xml_path, agreement_threshold=0):
    pos_lines = []
    neg_lines = []
    extended_lines = []

    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()

    xml = BeautifulSoup(markup, features="xml")

    if xml.LidcReadMessage is None:
        return None, None, None

    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    
    if patient_id not in patients:
        return None, None, None

    print(patient_id)

    image_path = patients[patient_id]
    itk_img = SimpleITK.ReadImage(image_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    rescale = spacing / settings.TARGET_VOXEL_MM

    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        # print("Sesion")
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)


        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            z_center -= origin[2]
            z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)

        pos_lines = filtered_lines

    df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",patient_id + "_annos_pos_lidc.csv")
    df_annos.to_csv(save_path, index=False)
    df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    save_path = os.path.join(settings.PATIENTS_DIR,"labels",patient_id + "_annos_neg_lidc.csv")
    df_neg_annos.to_csv(save_path, index=False)
    return pos_lines, neg_lines, extended_lines


def process_lidc_annotations(xml_dir, agreement_threshold=0):

    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []
    patients = utils.load_patients_list()
    for anno_dir in os.listdir(xml_dir):
        anno_subdir = os.path.join(xml_dir,anno_dir)
        xml_paths = [x for x in os.listdir(anno_subdir) if x[-4:] == ".xml"]
        for xml_file in xml_paths:
            xml_path = os.path.join(anno_subdir,xml_file)
            print(file_no, ": ",  xml_path)
            pos, neg, extended = load_lidc_xml(patients, xml_path, agreement_threshold)
            if pos==None:
                continue
            pos_count += len(pos)
            neg_count += len(neg)
            print("Pos: ", pos_count, " Neg: ", neg_count)
            file_no += 1
            all_lines += extended


    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    save_path = os.path.join(settings.BASE_DATA_DIR,"CSVFILES","lidc_annotations.csv")
    df_annos.to_csv(save_path, index=False)

def main():
    xml_dir = os.path.join(settings.BASE_DATA_DIR,"LIDC")
    process_lidc_annotations(xml_dir, agreement_threshold=0)

if __name__=="__main__":
    main()    
