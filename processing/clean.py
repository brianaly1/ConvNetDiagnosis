import os
import sys

file_dir = "/media/ubuntu/cryptscratch/scratch/alyb/Data/TrainData/LIDC/"
files = [os.path.join(file_dir,x) for x in os.listdir(file_dir)]
mal=[]

for file_name in files:
    if file_name[-12:-11] == "_":
        mal.append(int(file_name[-11:-10]))
    else:
        mal.append(int(file_name[-12:-10]))

for index,file_name in enumerate(files):
    if mal[index] == 9 or mal[index] == 4 or mal[index] == 1:
        os.remove(file_name)
