import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json

def read_json(fname):
    with open(fname, 'r') as fin:
        data = json.load(fin)
    return data

root_path = "/home/bakrinski/nobackup/segtool/RUNS/baseline/"
datasets = os.listdir(root_path)

metric = "Fscore"
# metric = "Weighted mean of: dice_loss and jaccard_loss)"

for dataset in datasets[:1]:
    decoders = os.listdir(root_path+"/"+dataset)
    for decoder in decoders:
        encoders = os.listdir(root_path+"/"+dataset+"/"+decoder)
        for encoder in encoders:
            folders = os.listdir(root_path+"/"+dataset+"/"+decoder+"/"+encoder)
            if("graphics" in folders):
                folders.remove("graphics")
            # print(folders)
            x_train = []
            x_valid = []
            for fold in folders:
                # data = json.load("train_logs.json")
                data = read_json(root_path+"/"+dataset+"/"+decoder+"/"+encoder+"/"+fold+"/train_logs.json")

                x_fold_train = []
                x_fold_valid = []
                for epoch in range(len(data["train"])):
                    x_fold_train.append(data["train"][epoch][metric])
                for epoch in range(len(data["valid"])):
                    x_fold_valid.append(data["valid"][epoch][metric])

                x_train.append(x_fold_train)
                x_valid.append(x_fold_valid)

            x_train = np.array(x_train)
            x_train = np.mean(x_train,axis=0)

            x_valid = np.array(x_valid)
            x_valid = np.mean(x_valid,axis=0)

            title = dataset+" "+decoder+" "+encoder
            print(title)

            # plt.title(title)
            plt.title("train validation curve "+dataset)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            #plt.ylim(0.0,1.0)
            plt.yscale("log")
            plt.plot(np.arange(len(x_train)), x_train)
            #plt.plot(np.arange(len(x_valid)), x_valid, linestyle='dashed')
            # plt.show()
            plt.tight_layout()
            plt.savefig("test2.pdf")


# plt.yscale("log")
# plt.plot(a)
# plt.plot(np.arange(len(a))*50, a)
# plt.show()
