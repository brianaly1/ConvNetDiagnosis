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

def worldToVox(centroids_world,spacing,scanOrigin):
    '''
    convert centroids from world frame (extracted from csv files, into voxel frame
    Inputs: 
        centroids_world : list of centroids in world coordinates 
        spacing: list of voxel spacings in mm 
        scanOrigin: location of voxel frame origin wrt world frame
    Outputs:
        centroids_vox: list of centroids in voxel coordinates
    note: the following only works if the voxel frame is not rotated with respect to the world frame - which is the case
    for all the data used
    '''
    centroids_world = np.array(centroids_world).astype(float)
    origin = np.array(scanOrigin).astype(float)
    scaled_vox = np.absolute(centroids_world - origin) #broadcast
    centroids_vox = scaled_vox / spacing.reshape((1,3))
    centroids_vox = centroids_vox.astype(int)
    return centroids_vox.tolist()	

def extractCandidate(volume,cen_vox,vox_size,spacing,new_spacing,translate): 
    '''
    given a centroid location in the volume, calculate the sub volume shape (old_shape) that would result
    in the desired sub volume shape after resampling at a new spacing
    Inputs: 
        volume: scan volume
        cen_vox: centroid in voxel coordinates
        spacing: list of original scan voxel spacings
        new_spacing: list of desired voxel spacings
        translations: list of translations for augmentation
    Outputs: 
        sub_vols: list of extracted sub volumes, 
        patches : 2d plane centered at centroid, used for visualization
    '''
    sub_vols = []
    patches = []
    resize_factor = spacing/new_spacing
    old_shape = np.round(vox_size/resize_factor)
    resize_factor = vox_size/old_shape
    new_spacing = spacing/resize_factor
    top_left = (cen_vox - old_shape/2).astype(int) 
    bot_right = (cen_vox + old_shape/2).astype(int) 
    sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
    sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
    assert np.shape(sub_volume) == (32,32,32) # using this as a lazy way to reject extracted sub volumes that dont fall within the original volume's bounds
    patch = sub_volume[int(vox_size[0]/2),:,:]
    sub_vols.append(sub_volume)
    patches.append(patch)
    if translate!=0:
        translations = np.random.randint(-SUBVOL_DIM[0]/4,high=SUBVOL_DIM[0]/4,size=(translate,1,3)) 
        for translation in translations:
            trans_np = np.squeeze(translation)
            top_left = (cen_vox - old_shape/2).astype(int) + trans_np
            bot_right = (cen_vox + old_shape/2).astype(int) + trans_np
            sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
            sub_volume = scipy.ndimage.interpolation.zoom(sub_volume, resize_factor, mode='nearest')
            assert np.shape(sub_volume) == (32,32,32) # using this as a lazy way to reject extracted sub volumes that dont fall within the original volume's bounds
            patch = sub_volume[int(vox_size[0]/2-trans_np[0]),:,:]
            sub_vols.append(sub_volume)
            patches.append(patch)
    return sub_vols, patches

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
            if np.any(np.absolute(cen_pos-cen)<SUBVOL_DIM) == True or np.any((cen-old_shape/2) < 0) == True or np.any((cen+old_shape/2) > vol_dim) == True:
                cen_rand.pop()
                break
    return cen_rand
    
def loadLUNASub(patients,series_uids,is_pos,save_path):
    count = 0
    if is_pos:
        sample_spacings = SPACINGS_POS
        translate = 19
        candidates = extract.getNodules()
    else:
        sample_spacings = SPACINGS_NEG
        translate = 0
        candidates = extract.getCandidates()
    with open(save_path,"wb") as openfile:
        for index,patient in enumerate(patients):
            series_uid = series_uids[index]
            try: 
                if series_uid in candidates:
                    image,origin,spacing = extract.load_itk_image(patient)    
                    patient_pixels = extract.getPixels(0,image,0)
                    centroids_world = candidates[series_uid]
                    centroids_vox = worldToVox(centroids_world,spacing,origin)
                    for centroid in centroids_vox:
                        for new_spacing in sample_spacings:    
                            try:    
                                sub_vols,patches = extractCandidate(patient_pixels,centroid,SUBVOL_DIM,spacing,new_spacing,translate)
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
                                        
def loadRandomSub(patients,series_uids,save_path):
    count = 0
    sample_spacings = SPACINGS_NEG
    candidates = extract.getNodules()
    translate = 0
    with open(save_path,"wb") as openfile:
        for index,patient in enumerate(patients):
            series_uid = series_uids[index]
            try:
                if series_uid in candidates:
                    image,origin,spacing = extract.load_itk_image(patient)
                    patient_pixels = extract.getPixels(0,image,0)
                    centroids_world = candidates[series_uid]
                    centroids_vox = worldToVox(centroids_world,spacing,origin)
                    for new_spacing in sample_spacings:
                        try:
                            vol_dim = np.array(np.shape(patient_pixels))
                            centroids_rand = getRandom(vol_dim,centroids_vox,SUBVOL_DIM,spacing,new_spacing,25)
                            for centroid in centroids_rand:
                                sub_vols,patches = extractCandidate(patient_pixels,centroid,SUBVOL_DIM,spacing,new_spacing,translate)    
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

def loadLunaVol(patients,series_uids,save_path,desired_positives,desired_negatives):
    ''' 
    load a few full volumes to test the first stage of the pipeline
    save a list of volumes and the locations of the nodules in each
    Inputs:
        save_path: path to save the pickled volumes
        luna_dir: directory containing the luna files
    '''
    candidates = extract.getNodules()
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
                    centroids_vox = worldToVox(centroids_world,spacing,origin)
                    pickle.dump([patient_pixels,centroids_vox],openfile)
                    pos_count += 1
                elif series_uid not in candidates and neg_count<desired_negatives:
                    pickle.dump([patient_pixels,None],openfile)
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
    loadLunaVol(patients,series_uids,VOL_PATH,desired_positives = 100, desired_negatives = 100)
    

if __name__ == "__main__":
    main()
