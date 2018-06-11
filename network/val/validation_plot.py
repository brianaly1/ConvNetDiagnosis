import pickle
from matplotlib import pyplot as plt

VAL_FILE = "/home/alyb/trained_models/val/validation.p"

def plot_graph(path):
    with open(path,'rb') as openfile:
        val = pickle.load(openfile)
    loss = val["accs"]
    plt.plot(loss)
    plt.ylabel("accuracy")
    plt.xlabel("iteration (per 90)")
    plt.show()    

def main():
    plot_graph(VAL_FILE)

if __name__=="__main__":
    main()  
