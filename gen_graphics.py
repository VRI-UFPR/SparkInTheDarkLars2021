import os
import cv2
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt

def getminmax(x1, x2):
    aux1 = sorted(x1)
    aux2 = sorted(x2)

    #print(aux1[0], aux2[0])
    if aux1[0] < aux2[0]:
        p = aux1[0]
    else:
        p = aux2[0]
    
    #print(p)
    p = p * 10
    #print(p)
    if p < 0:
        p = 0
    #print('----------')
    if aux1[len(aux1)-1] > aux2[len(aux2)-1]:
        e = aux1[len(aux1)-1]
    else:
        e = aux2[len(aux2)-1]
    #print(e)
    e = e * 10 + 1
    #print(e)
    if e > 10:
        e = 10
    #print(math.floor(p)*10, int(round(e,0))*10+5)
    return math.floor(p)*10, int(round(e,0))*10+5

if __name__ == '__main__':

    experiments = ['baseline/']
    
    encoders = ['resnet18', 
            'resnet34', 
            'resnet50', 
            'resnet101', 
            'resnet152', 
            'resnext50_32x4d', 
            'resnext101_32x8d',
            'timm-resnest14d', 
            'timm-resnest26d',
            'timm-resnest50d', 
            'timm-resnest101e',
            'timm-resnest200e', 
            'timm-resnest269e',
            'timm-resnest50d_4s2x40d', 
            'timm-resnest50d_4s2x40d',
            'timm-res2net50_26w_4s', 
            'timm-res2net101_26w_4s',
            'timm-res2net101_26w_4s', 
            'timm-res2net50_26w_8s',
            'timm-res2net50_48w_2s', 
            'timm-res2net50_14w_8s',
            'timm-res2next50',
            'timm-regnetx_002',
            'timm-regnetx_004', 
            'timm-regnetx_006',
            'timm-regnetx_008', 
            'timm-regnetx_016',
            'timm-regnetx_032', 
            'timm-regnetx_040',
            'timm-regnetx_064', 
            'timm-regnetx_080',
            'timm-regnetx_120', 
            'timm-regnetx_160',
            'timm-regnetx_320',
            'timm-regnety_002',
            'timm-regnety_004', 
            'timm-regnety_006',
            'timm-regnety_008', 
            'timm-regnety_016',
            'timm-regnety_032', 
            'timm-regnety_040',
            'timm-regnety_064', 
            'timm-regnety_080',
            'timm-regnety_120', 
            'timm-regnety_160',
            'timm-regnety_320',
            'senet154',
            'se_resnet50', 
            'se_resnet101',
            'se_resnet152', 
            'se_resnext50_32x4d',
            'se_resnext101_32x4d', 
            'timm-skresnet18',
            'timm-skresnet34', 
            'timm-skresnext50_32x4d',
            'densenet121', 
            'densenet169',
            'densenet201', 
            'densenet161',
            'inceptionresnetv2', 
            'inceptionv4',
            'xception', 
            'efficientnet-b0',
            'efficientnet-b1', 
            'efficientnet-b2',
            'efficientnet-b3', 
            'efficientnet-b4',
            'efficientnet-b5', 
            'efficientnet-b6',
            'efficientnet-b7', 
            'timm-efficientnet-b0',
            'timm-efficientnet-b1', 
            'timm-efficientnet-b2',
            'timm-efficientnet-b3', 
            'timm-efficientnet-b4',
            'timm-efficientnet-b5', 
            'timm-efficientnet-b6',
            'timm-efficientnet-b7', 
            'timm-efficientnet-b8',
            'timm-efficientnet-l2', 
            'timm-efficientnet-lite0',
            'timm-efficientnet-lite1', 
            'timm-efficientnet-lite2',
            'timm-efficientnet-lite3',
            'timm-efficientnet-lite4', 
            'mobilenet_v2',
            'dpn68', 
            'dpn68b',
            'dpn92', 
            'dpn98',
            'dpn107', 
            'dpn131',
            'vgg11', 
            'vgg11_bn',
            'vgg13', 
            'vgg13_bn',
            'vgg16', 
            'vgg16_bn',
            'vgg19', 
            'vgg19_bn']

    encoders = ['resnet50/','resnet101/','resnext50_32x4d/','resnext101_32x8d/',
                'timm-res2net50_26w_4s/','timm-res2net101_26w_4s/','vgg16/','densenet121/',
                'densenet169/','densenet201/']
    decoders = ['unetplusplus/', 'unet/','fpn/','pspnet/','linknet/', 'manet/']
    #decoders = ['unet/']
    datasets = ['medseg/', 'covid19china/', 'mosmed/', 'covid20cases/','ricord1a/']
    #datasets = ['covid19china/']
    runs_path = 'RUNS/'

    for experiment in experiments:
        for dataset in datasets:
            for decoder in decoders:
                for encoder in encoders:
                    runspath = runs_path + experiment + dataset + decoder + encoder
                    
                    mean_results_path = runspath + '/graphics'
                    if os.path.isdir(mean_results_path):
                        os.system('rm -rf {}'.format(mean_results_path))

                    runs = glob.glob(runspath + '*')

                    if len(runs) != 5:
                        print('ERROR!!!!')
                        print(runs)
                        exit()

                    all_train = []
                    all_valid = []
                    num_classes = 0
                    labels = []

                    for run in runs:
                        print(run)
                        results_path = run + '/graphics'
                        if os.path.isdir(results_path):
                            os.system('rm -rf {}'.format(results_path))

                        os.mkdir(results_path)

                        train_logs_path = run + '/train_logs.json'

                        with open(train_logs_path) as train_logs_file:
                            train_logs = json.load(train_logs_file)
                        
                        if len(labels) == 0:
                            epoch_0 = train_logs['train'][0]
                            for key, value in epoch_0.items():
                                labels.append(key)
                            num_classes = len(labels) - 3

                        train_results = train_logs['train']
                        valid_results = train_logs['valid']
                        #print(labels)
                        for label in labels:
                            if label != 'Time' and label != 'Epoch' and label != 'Weighted mean of: dice_loss and jaccard_loss)':
                                print(label)
                                x1 = []
                                x2 = []
                                for train_epoch, valid_epoch in zip(train_results, valid_results):
                                    x1.append(train_epoch[label])
                                    x2.append(valid_epoch[label])

                                #if label == 'fscore':
                                #    print(x1)
                                
                                all_train.append(x1)
                                all_valid.append(x2)
                                
                                y = list(range(len(x1)))
                                
                                p, e = getminmax(x1, x2)
                                #print(int(p)*10)
                                #print(int(e)*10+1)
                                yticks = [p/100 for p in range(p, e, 5)]
                                xticks = [p for p in range(0, len(x1)+5, 5)]
                                plt.plot(y, x1, label='train')
                                plt.plot(y, x2, label='valid')
                                plt.xticks(xticks)
                                plt.yticks(yticks)
                                plt.title(label)
                                plt.legend()
                                plt.grid(True)
                                plt.savefig(results_path + '/' + label.lower() + '.png')
                                plt.clf()
                    print("--------------------------------------------------")
                    os.mkdir(mean_results_path)
                    
                    for i in range(num_classes):
                        mean_train = []
                        mean_valid = []
                        print(labels[i+1])
                        for j in range(0+i, len(all_train), num_classes):
                            mean_train.append(all_train[j])
                            mean_valid.append(all_valid[j])

                        mean_train = list(map(lambda x: sum(x)/len(x), zip(*mean_train)))
                        mean_valid = list(map(lambda x: sum(x)/len(x), zip(*mean_valid)))

                        p, e = getminmax(mean_train, mean_valid)

                        y = list(range(len(mean_train)))
                        yticks = [p/100 for p in range(p, e, 5)]
                        xticks = [p for p in range(0, len(mean_train)+5, 5)]
                        plt.plot(y, mean_train, label='train')
                        plt.plot(y, mean_valid, label='valid')
                        plt.xticks(xticks)
                        plt.yticks(yticks)
                        plt.title(dataset + experiment.split('_')[0] + labels[i+1])
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(mean_results_path + '/' + labels[i+1].lower() + '.png')
                        plt.clf()
