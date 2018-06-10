import numpy as np
import pickle
import visualize
import cv2
import matplotlib.pyplot as plt

def slice_volume(patient_dic):
    '''
    split patient volume into slices in all three dimensions
    input: dictionary with patient ID as keys and lists of [volumes,scales]
    ouput: list of slices 
    '''
    patients = patient_dic.keys()
    volumes = []
    slices = []
    min_depth = 1000
    num_slices = 0
    for patient in patients:
        volume = patient_dic[patient][0]
        volumes.append(volume)
        dimensions = volume.shape
        for k in range (0,dimensions[0]):
            slices.append(volume[k,:,:])
            num_slices = num_slices + 1
        for i in range (0,dimensions[1]):
            slices.append(volume[:,i,:])
            num_slices = num_slices + 1
        for j in range (0,dimensions[2]):
            slices.append(volume[:,:,j])
            num_slices = num_slices + 1
    return slices

def label_slices(slices,px_size):
    '''
    bulk labelling and resizing of slices
    input: slices, desired rescaled slice size
    ouput: list of resized slices, list of labels
    '''
    count = 0
    num_slices = len(slices)
    labels = []
    resized_imgs = []
    while count < num_slices:
        while True:
            try:
                quantity = input("how many images would you like to label at once? : ")
                if quantity.isnumeric() == False:
                    raise Exception('enter a number')
                else:
                    quantity = int(quantity)
                if quantity <= 0:
                    raise Exception('greater than zero')
                elif quantity > 16:
                    raise Exception('less than seventeen')
                else:
                    break
            except Exception as error:
                print('Error: ' + repr(error))

        fig = plt.figure()
        for num,each_slice in enumerate(slices[count:count+quantity]):
            sub_plot = fig.add_subplot(4,4,num+1)
            resized_img = cv2.resize(np.array(each_slice),(px_size,px_size))
            resized_imgs.append(resized_img)
            sub_plot.imshow(resized_img)
        plt.show() 

        while True: 
            try:
                label = input("Do these images contain lungs? (y/n) : ")
                if label != "y" and label != "n" and label != "b":
                    raise Exception('invalid label')
                if label == 'y':
                    labels.extend([1]*quantity)
                    count = count + quantity
                    break
                elif label == 'n':
                    labels.extend([0]*quantity)
                    count = count + quantity
                    break
                else: # 'b' lets you re-do the current iteration with a possibly different quantity
                    break 
            except Exception as error:
                print('Error: ' + repr(error))

        plt.close()
        print('count is: ')
        print(count)
        print('labels are: ')
        print(labels)
        with open("slices.p","wb") as openfile:
            pickle.dump(resized_imgs,openfile)
        with open("labels.p","wb") as openfile:
            pickle.dump(labels,openfile)
        print('saved...')
    return 1

def main():
    with open("dataset.p","rb") as openfile:
        patient_dic = pickle.load(openfile)
    slices = slice_volume(patient_dic)
    labelled = label_slices(slices,128)
    print(labelled)

main()
