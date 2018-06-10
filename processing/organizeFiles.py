import os
import shutil
import sys
import argparse

# read scan directory
parser = argparse.ArgumentParser(description = 'Dicom preprocess')
parser.add_argument('--scan-dir', type=str, default='/media/brian/Data2/Ubuntu/LIDC-IDRI/', metavar='N',help='input path to directory containing scans(grouped in folders')
args = parser.parse_args()

INPUT_FOLDER = args.scan_dir
patients = os.listdir(INPUT_FOLDER)
patients.sort()

def organize(path):
    '''
    reorganize downloaded LIDC data into a single directory per scan
    inputs: path to LIDC folder
    '''
    sub_dirs = os.listdir(path)
    if len(sub_dirs) == 2:
        subdir1 = sub_dirs[0]
        del_subdir = subdir1
        subdir2 = sub_dirs[1]
        len_del1 = len(os.listdir(path+subdir1+'/'+(os.listdir(path+subdir1)[0])))
        len_del2 = len(os.listdir(path+subdir2+'/'+(os.listdir(path+subdir2)[0])))
        if len_del2 < len_del1:
            del_subdir = subdir2
        shutil.rmtree(path+'/'+del_subdir)
    sub_dirs = os.listdir(path)
    if len(sub_dirs) == 1:
        file_dir = path + sub_dirs[0] + '/' + (os.listdir(path+sub_dirs[0])[0])
        for this_file in os.listdir(file_dir):
            if this_file[-4:] == '.xml':
                os.remove(file_dir + '/' + this_file)
            else:
                shutil.copyfile(file_dir + '/' + this_file, path+'/'+this_file)
        shutil.rmtree(path + '/' + sub_dirs[0])            
    return 1

def main():
    for patient in patients:
        organize(INPUT_FOLDER + patient+'/')
main()
