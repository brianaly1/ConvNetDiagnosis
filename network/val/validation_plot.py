import pickle
from matplotlib import pyplot as plt

VAL_FILE = "/media/ubuntu/cryptscratch/scratch/alyb/Data/Cancer/TrainData/Val/roi/validation.p"

def plot_graph(path):
    with open(path,'rb') as openfile:
        val = pickle.load(openfile)
    loss = val["accs"]
    plt.plot(loss)
    plt.ylabel("acc")
    plt.xlabel("iteration group")
    plt.show()    

def main():
    plot_graph(VAL_FILE)

if __name__=="__main__":
    main()  
