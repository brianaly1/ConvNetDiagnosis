import pickle
from matplotlib import pyplot as plt

VAL_FILE = "/home/alyb/ConvNetDiagnosis/network/val/validation.p"

def plot_graph(path):
    with open(path,'rb') as openfile:
        val = pickle.load(openfile)
    loss = val["loss"]
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("iteration group")
    plt.show()    

def main():
    plot_graph(VAL_FILE)

if __name__=="__main__":
    main()  
