import sys
import os
import utils
import settings
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


CENS = []
NODS = []

def prev_slice(ax):
    patch_list = ax.patches
    [p.remove() for p in patch_list]
    volume = ax.volume
    sub_vol_size = settings.SUB_VOL_SHAPE[0]
    ax.index = (ax.index-1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    for cen in CENS:
        lower_bound = int(cen[2]-sub_vol_size/2)
        upper_bound = int(cen[2]+sub_vol_size/2)
        if ax.index >= lower_bound and ax.index <= upper_bound:
            bot_left = (cen[0]-sub_vol_size/2,cen[1]+sub_vol_size/2)
            w = sub_vol_size
            h = sub_vol_size 
            rect = patches.Rectangle(bot_left,w,h,linewidth=2,edgecolor='r',fill=False)
            ax.add_patch(rect)
    for cen in NODS:
        lower_bound = int(cen[2]-sub_vol_size/2)
        upper_bound = int(cen[2]+sub_vol_size/2)
        if ax.index >= lower_bound and ax.index <= upper_bound:
            bot_left = (cen[0]-sub_vol_size/2,cen[1]+sub_vol_size/2)
            w = sub_vol_size
            h = sub_vol_size 
            rect = patches.Rectangle(bot_left,w,h,linewidth=2,edgecolor='g',fill=False)
            ax.add_patch(rect)

def next_slice(ax):
    patch_list = ax.patches
    [p.remove() for p in patch_list]
    volume = ax.volume
    sub_vol_size = settings.SUB_VOL_SHAPE[0]
    ax.index = (ax.index+1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    for cen in CENS:
        lower_bound = int(cen[2]-sub_vol_size/2)
        upper_bound = int(cen[2]+sub_vol_size/2)
        if ax.index >= lower_bound and ax.index <= upper_bound:
            bot_left = (cen[0]-sub_vol_size/2,cen[1]+sub_vol_size/2)
            w = sub_vol_size
            h = sub_vol_size 
            rect = patches.Rectangle(bot_left,w,h,linewidth=2,edgecolor='r',fill=False)
            ax.add_patch(rect)
    for cen in NODS:
        lower_bound = int(cen[2]-sub_vol_size/2)
        upper_bound = int(cen[2]+sub_vol_size/2)
        if ax.index >= lower_bound and ax.index <= upper_bound:
            bot_left = (cen[0]-sub_vol_size/2,cen[1]+sub_vol_size/2)
            w = sub_vol_size
            h = sub_vol_size 
            rect = patches.Rectangle(bot_left,w,h,linewidth=2,edgecolor='g',fill=False)
            ax.add_patch(rect)

def process_key(event):  
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'a':
        prev_slice(ax)
    elif event.key == 'd':
        next_slice(ax)
    fig.canvas.draw()

def visSlice(volume):
    volume = np.array(volume)
    plt.ion()
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event',process_key)
    plt.show()
    _ = input("Press [Enter] for next example.")
    plt.close()
    
def view_candidates(path,centroids):
    '''
    show patient volume with rectangles around centroids
    Inputs:
        path: path to patient test file
        centroids: list of candidates
    '''
    uid = path.split("/")[-1]
    uid = uid[0:-10]
    volume = utils.load_patient_images(uid, "_i.png")
    nods = utils.get_patient_nodules(uid)
    NODS.extend(nods) 
    CENS.extend(centroids)
    print("Centroids of interest: {}".format(CENS))
    print("Nodule locations:: {}".format(NODS))
    visSlice(volume)

