import sys
import os
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/processing/')
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/network/')
import c3d
import c3d_predict
import c3d_train
import process_train
import process_test
import shutil
import process_ndsb

def main():
    #process_train.ndsb_main()
    c3d_train.main(mode=1)
    

        
if __name__=="__main__":
    main()
