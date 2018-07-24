import sys
import os
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/processing/')
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/network/')
import settings
import process_luna
import process_ndsb
import process_lidc
import process_annotations

def main():

    '''
    This script walks through the entire pipeline.
    Note that some manual intervention is required,
    the process has not been entirely automated.
    '''

    #Update settings.py file based on your workstation and preferences
    #Create relevant directories
    dirs = [settings.CA_RAW_DATA_DIR, settings.CA_PRC_DATA_DIR, settings.CA_TRAIN_DATA_DIR, settings.CA_TEST_RES_DIR]
    [os.mkdir(dir_path) for dir_path in dirs if not os.path.exists(dir_path)] 

    #Download LUNA data, NDSB Stage 1 data, and LIDC annotations and place in raw data directory according to the layout in settings.py

    #Process raw luna images into png files and save to processed images directory
    dirs = [settings.CA_LUNA_IMG_DIR]
    [os.mkdir(dir_path) for dir_path in dirs if not os.path.exists(dir_path)]     
    #process_luna.main() 

    #Process raw ndsb dicom images into png files and save to processed images directory
    dirs = [settings.CA_NDSB_IMG_DIR]
    [os.mkdir(dir_path) for dir_path in dirs if not os.path.exists(dir_path)]     
    #process_ndsb.main() 

    #Process LIDC xml annotations, and save csv files with mined nodule locations
    dirs = [settings.CA_LABELS_DIR]
    [os.mkdir(dir_path) for dir_path in dirs if not os.path.exists(dir_path)]  
    #process_lidc.main()

    #Process LUNA and Auto generated annotations
    #process_annotations.main()

    #Extract and process nodules from processed LUNA images and LUNA/LIDC/AUTO annotations
     

    #process_test.ndsb_main(mag=2)
    #c3d_predict.main(mag=2)

        
if __name__=="__main__":
    main()
