import numpy as np
import pickle
import visualize
import sys
import os
import extract
import scipy.ndimage
#sys.path.insert(0,'/home/alyb/ConvNetDiagnosis/network')
#import dataset

LUNA_DIR = "/home/alyb/data/luna16/" 
SUBVOL_DIM = np.array([32,32,32]) # extracted sub volumes will have these dimensions (z,y,x)
SPACINGS_POS = [[1.0,1.0,1.0],[0.5,0.5,0.5],[2.0,2.0,2.0]] #augmentation for positive labels - extract sub volumes at various scales
SPACINGS_NEG = [[1.0,1.0,1.0]]

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
    spacing = np.array(spacing).astype(float)
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
        translate: augment with 20 translations or not
    Outputs: 
        sub_vols: list of extracted sub volumes, 
        patches : 2d plane centered at centroid, used for visualization
    '''
    sub_vols = []
    patches = []
    top_left = (cen_vox - SUBVOL_DIM/2).astype(int) 
    bot_right = (cen_vox + SUBVOL_DIM/2).astype(int) 
    sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
    assert np.shape(sub_volume) == (32,32,32) # using this as a lazy way to reject extracted sub volumes that dont fall within the original volume's bounds
    patch = sub_volume[int(vox_size[0]/2),:,:]
    sub_vols.append(sub_volume)
    patches.append(patch)
    if translate!=0:
        translations = np.random.randint(-SUBVOL_DIM[0]/4,high=SUBVOL_DIM[0]/4,size=(translate,1,3)) 
        for translation in translations:
            trans_np = np.squeeze(translation)
            top_left = (cen_vox - SUBVOL_DIM/2).astype(int) + trans_np
            bot_right = (cen_vox + SUBVOL_DIM/2).astype(int) + trans_np
            sub_volume = volume[top_left[0]:bot_right[0],top_left[1]:bot_right[1],top_left[2]:bot_right[2]]
            assert np.shape(sub_volume) == (32,32,32) 
            patch = sub_volume[int(vox_size[0]/2-trans_np[0]),:,:]
            sub_vols.append(sub_volume)
            patches.append(patch)
    return sub_vols, patches

def getRandom(vol_dim,cen_vox,quantity):
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
    while len(cen_rand) < quantity:
        cen = (np.random.rand(3) * vol_dim).astype(int)
        cen_rand.append(cen)
        for cen_pos in cen_vox:
            if np.any(np.absolute(cen_pos-cen)<SUBVOL_DIM) == True or np.any((cen-16) < 0) == True or np.any((cen+16) > vol_dim) == True:
                cen_rand.pop()
                break
    return cen_rand
    
def loadLUNA(pos,path):
    count = 0
    if pos==1:
        sample_spacings = SPACINGS_POS
        translate = 19
        candidates = extract.getNodules()
    else:
        sample_spacings = SPACINGS_NEG
        translate = 0
        candidates = extract.getCandidates()
    with open(path,"wb") as openfile:
        subsets = os.listdir(LUNA_DIR)
        for subset in subsets:
            patients = os.listdir(LUNA_DIR + subset + "/")
            for patient in patients:
                #try: 
                if patient[-4:] == ".mhd": 
                    series_uid = patient[:-4]
                    if series_uid in candidates:
                        image,origin,spacing = extract.load_itk_image(LUNA_DIR + subset + "/" + patient)    
                        patient_pixels = extract.getPixels(0,image,0)                   
                        centroids_world = candidates[series_uid]
                        for new_spacing in sample_spacings: 
                            centroids_vox = worldToVox(centroids_world,new_spacing,origin)  
                            patient_pixels = extract.resample(patient_pixels,spacing,new_spacing)  
                            for centroid in centroids_vox:   
                                #try:             
                                sub_vols,patches = extractCandidate(patient_pixels,centroid,SUBVOL_DIM,spacing,new_spacing,translate)
                                count = count+len(sub_vols)                            
                                pickle.dump([sub_vols,patches,series_uid],openfile)
                                #    except AssertionError:
                                #        print("sub volume extraction error: most likely due to out of range index")
                                #    except KeyboardInterrupt:
                                #        print("Interrupted")
                                #        sys.exit()
                                #    except:
                                #        print("unknown error with candidate sub volume")
                            #print(str(count) + " sub volumes saved")
                    else:
                        print("Patient: " + series_uid + " not in candidates")

                #except KeyboardInterrupt:
                #    print("Interrupted")
                #    sys.exit()
                #except:
                #    print("unkown error with patient: " + patient)
                                         
def loadRandom(path):
    count = 0
    sample_spacings = SPACINGS_NEG
    candidates = extract.getNodules()
    translate = 0
    with open(path,"wb") as openfile:
        subsets = os.listdir(LUNA_DIR)
        for subset in subsets:
            patients = os.listdir(LUNA_DIR + subset + "/")
            for patient in patients:
                try:
                    if patient[-4:] == ".mhd":
                        series_uid = patient[:-4]
                        if series_uid in candidates:
                            image,origin,spacing = extract.load_itk_image(LUNA_DIR + subset + "/" + patient)
                            patient_pixels = extract.getPixels(0,image,0)
                            centroids_world = candidates[series_uid]
                       
                            for new_spacing in sample_spacings:
                                patient_pixels = extract.resample(patient_pixels,spacing,new_spacing)
                                centroids_vox = worldToVox(centroids_world,new_spacing,origin)
                                vol_dim = np.array(np.shape(patient_pixels))
                                try:                                                     
                                    centroids_rand = getRandom(vol_dim,centroids_vox,25)
                                    for centroid in centroids_rand:
                                        sub_vols,patches = extractCandidate(patient_pixels,centroid,SUBVOL_DIM,spacing,new_spacing,translate)    
                                        count = count+len(sub_vols)
                                        pickle.dump([sub_vols,patches,series_uid],openfile)
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


def main():
    pos_path = '/home/alyb/data/pickles/nodules.p'
    loadLUNA(1,pos_path)
    falsepos_path = '/home/alyb/data/pickles/false_pos.p'
    loadLUNA(0,falsepos_path)
    random_path = '/home/alyb/data/pickles/random.p'
    loadRandom(random_path)
    #save_dir = '/home/alyb/data/tfrecords/'
    #dataset.createDataset(pos_path,falsepos_path,random_path,save_dir)

if __name__ == "__main__":
    main()
