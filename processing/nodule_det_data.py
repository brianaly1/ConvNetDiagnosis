import numpy as np
import pickle
import visualize
import sys
import os
import extract
import scipy.ndimage
#sys.path.insert(0,'/home/alyb/ConvNetDiagnosis/network')
#import dataset
import random

LUNA_DIR = "/home/alyb/data/luna16/" 
SUBVOL_DIM = np.array([32,32,32]) # extracted sub volumes will have these dimensions (z,y,x)
SPACINGS_POS = [[1.5,0.5,0.5],[1.375,0.625,0.625],[1.25,0.75,0.75]] #augmentation for positive labels - extract sub volumes at various scales
SPACINGS_NEG = [[1.25,0.75,0.75]]
POS_PATH = '/home/alyb/data/pickles/nodules.p'
FALSE_POS_PATH = '/home/alyb/data/pickles/false_pos.p'
RANDOM_PATH = '/home/alyb/data/pickles/random.p'
TFR_DIR = '/home/alyb/data/tfrecords/'
VOL_PATH = '/home/alyb/data/pickles/volumes.p'
CSV_FILES = '/home/alyb/data/CSVFILES/annotations.csv'

def getRandom(vol_dim,cen_vox,vox_size,spacing,new_spacing,quantity):
    '''
    compute a list of random coordinates to extract negative sub volumes centered at the coordinates
    ensure the sub volume to be extracted is far enough from all positive sub volumes in the volume, and falls
    within the bounds of the volume, the volume to be extracted will have shape such that after resampling we get the desired sub volume shape
    Inputs: 
        vol_dim: scan volume dimensions
        cen_vox: list of positive centroids in this volume
        vox_size: desired sub volume size 
        spacing: list of original scan voxel spacings
        new_spacing: list of desired voxel spacings
        quantity: number of random sub volumes to extract
    Outputs: 
        cen_rand: list of centroids for random sub volumes
    '''
    cen_rand = []
    resize_factor = spacing/new_spacing
    old_shape = np.round(vox_size/resize_factor)
    resize_factor = vox_size/old_shape
    new_spacing = spacing/resize_factor
    while len(cen_rand) < quantity:
        cen = (np.random.rand(3) * vol_dim).astype(int)
        cen_rand.append(cen)
        for cen_pos in cen_vox:
            if np.all(np.absolute(cen_pos-cen)<vox_size) == True or np.any((cen-old_shape/2) < 0) == True or np.any((cen+old_shape/2) > vol_dim) == True:
                cen_rand.pop()
                break
    return cen_rand
    
def loadLUNASub(patients,series_uids,mode,save_path,csv_path):
    count = 0
    
    # mode 0 loads positive nodules, 1 false positives, 2 random non-nodules
    if mode==0:
        sample_spacings = SPACINGS_POS
        translate = 19
        candidates = extract.getCandidates(is_pos=1,file_path=csv_path)
    elif mode==1:
        sample_spacings = SPACINGS_NEG
        translate = 0
        candidates = extract.getCandidates(is_pos=0,file_path=csv_path)
    elif mode==2:
        sample_spacings = SPACINGS_NEG
        translate = 0
        candidates = extract.Candidates(is_pos=1,file_path=csv_path)

    with open(save_path,"wb") as openfile:
        for index,patient in enumerate(patients):
            series_uid = series_uids[index]
            try: 
                if series_uid in candidates:
                    image,origin,spacing = extract.load_itk_image(patient)    
                    patient_pixels = extract.getPixels(0,image,0)
                    centroids_world = candidates[series_uid]
                    centroids_vox = extract.worldToVox(centroids_world,spacing,origin)
                    centroids = centroids_vox
                    vol_dim = np.array(np.shape(patient_pixels))
                    for new_spacing in sample_spacings:  

                        if mode==2:
                            centroids_rand = getRandom(vol_dim,centroids_vox,SUBVOL_DIM,spacing,new_spacing,25)  
                            centroids = centroids_rand   
                   
                        for centroid in centroids:  
                            try:    
                                sub_vols,patches = extract.extractCandidate(patient_pixels,centroid,SUBVOL_DIM,spacing,new_spacing,translate)
                                count = count+len(sub_vols)                            
                                pickle.dump([sub_vols,patches,series_uid,new_spacing],openfile)
                            except AssertionError:
                                print("sub volume extraction error: most likely due to out of range index")
                            except KeyboardInterrupt:
                                print("Interrupted")
                                sys.exit()
                            except:
                                print("unknown error with candidate sub volume")

                    print(str(count) + " sub volumes saved")
                else:
                    print("Patient: " + series_uid + " not in candidates")
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()                    
            except:
                print("unkown error with patient: " + patient)
                                        
def loadLunaVol(patients,series_uids,save_path,desired_positives,desired_negatives,csv_path):
    ''' 
    load a few full volumes to test the first stage of the pipeline
    save a list of volumes and the locations of the nodules in each
    Inputs:
        save_path: path to save the pickled volumes
        luna_dir: directory containing the luna files
    '''
    candidates = extract.getCandidates(1,csv_path)
    with open(save_path,"wb") as openfile:
        indices = np.array(range(0,len(patients)))
        np.random.shuffle(indices)
        indices = indices.tolist() 
        pos_count = 0
        neg_count = 0
        patient_count = 0
        while pos_count < desired_positives or neg_count < desired_negatives:
            try:                  
                patient = patients[indices[patient_count]]   
                series_uid = series_uids[indices[patient_count]]  
                image,origin,spacing = extract.load_itk_image(patient)  
                patient_pixels = extract.getPixels(0,image,0) 
                if series_uid in candidates and pos_count<desired_positives:         
                    centroids_world = candidates[series_uid]
                    centroids_vox = extract.worldToVox(centroids_world,spacing,origin)
                    pickle.dump([patient_pixels,centroids_vox,spacing],openfile)
                    pos_count += 1
                elif series_uid not in candidates and neg_count<desired_negatives:
                    pickle.dump([patient_pixels,None,spacing],openfile)
                    neg_count += 1  
                patient_count += 1    
                print("positive count: %d, negative count: %d" %(pos_count,neg_count))  
            except IndexError:
                print("Iteration complete")
                sys.exit()
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()
            except:
                print("error extracting patient: %s" %(series_uid))    
                    
def main():
    subsets = os.listdir(LUNA_DIR)
    patients = []
    series_uids = []
    for subset in subsets:
        subset_path = os.path.join(LUNA_DIR,subset)
        subset_files = [x for x in os.listdir(subset_path) if x[-4:] == ".mhd"]   
        subset_series_uids = [x[:-4] for x in subset_files]   
        subset_patients = [os.path.join(subset_path,x) for x in subset_files]
        patients.extend(subset_patients)
        series_uids.extend(subset_series_uids)
    assert len(patients) == len(series_uids), "patients list and series uid list have different lengths!" 
    loadLunaVol(patients,series_uids,VOL_PATH,100,100,CSV_FILES)
    

if __name__ == "__main__":
    main()
