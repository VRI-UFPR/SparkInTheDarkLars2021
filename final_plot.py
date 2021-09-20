import os
import cv2
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

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

def plot(label, train, valid, save_path):
    y = list(range(len(train)))
    #p, e = getminmax(train, valid)
    p, e = 0, 100
    yticks = [p/100 for p in range(p, e, 5)]
    xticks = [p for p in range(0, len(train)+5, 5)]
    plt.plot(y, train, label='Train')
    plt.plot(y, valid, label='Valid')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(label)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + label.lower().replace(' ','_').replace(':','_') + '.pdf')
    plt.clf()

if __name__ == '__main__':

    root = 'RUNS/'

    super_runs = ['0p1/', '0p2/']
    super_runs = ['50e/']
    
    experiments = ['Clahe/', 'CoarseDropout/', 'ElasticTransform/', 'Emboss/', 'Flip/', 
                   'GaussianBlur/', 'GridDistortion/', 'GridDropout/', 'ImageCompression/', 'MedianBlur/',
                   'OpticalDistortion/', 'PiecewiseAffine/', 'Posterize/', 'RandomBrightnessContrast/', 'RandomCrop/',
                   'RandomGamma/', 'RandomSnow/', 'Rotate/', 'Sharpen/', 'ShiftScaleRotate/']
    experiments = ['noda/']
    #encoders = ['resnet50/','resnet101/','resnext50_32x4d/','resnext101_32x8d/',
    #            'timm-res2net50_26w_4s/','timm-res2net101_26w_4s/','vgg16/','densenet121/',
    #            'densenet169/','densenet201/', 'se_resnext50_32x4d/', 'se_resnext101_32x4d/',
    #            'se_resnet50/', 'se_resnet101/',
    #            'timm-regnetx_002/','timm-regnetx_004/','timm-regnetx_006/',
    #            'timm-regnety_002/','timm-regnety_004/','timm-regnety_006/']
    encoders = ['timm-regnetx_002/']

    #decoders = ['unetplusplus/', 'unet/','fpn/','pspnet/','linknet/', 'manet/']
    decoders = ['unetplusplus/']

    datasets = ['covid19china/', 'medseg/', 'mosmed/', 'ricord1a/', 'covid20cases/',]

    for super_run in super_runs:

        for experiment in experiments:

            runs_path = root + super_run + experiment

            for dataset in datasets:

                # MAKE A DATASET PLOT

                dataset_path = runs_path + dataset
                decoders_train = []
                decoders_valid = []
                #colors = ['red', 'purple', 'blue','orange','green', 'grey']
                colors = [[0, 0, 166], [255, 74, 70], [0, 137, 65], [153, 0, 153], [96, 96, 96], [255, 128, 0]]
                colors = np.array(colors)/255
                #colors = ['mediumblue', 'darkred', 'green', 'saddlebrown', 'dodgerblue', 'saddlebrown']

                decoders_train = []
                decoders_valid = []
                for decoder in decoders:

                    decoder_path = dataset_path + decoder 

                    for e, encoder in enumerate(encoders):

                        encoder_path = decoder_path + encoder

                        # MAKE THE MEAN CURVE OF 5 FOLDS

                        graphics_path = encoder_path + 'graphics/'
                        if os.path.isdir(graphics_path):
                            os.system('rm -rf {}'.format(graphics_path))

                        runs = glob.glob(encoder_path + '*')
                        for r, run in enumerate(runs):
                            print(run)

                            # READ RESULTS
                            train_logs_path = run + '/train_logs.json'
                            with open(train_logs_path) as train_logs_file:
                                train_logs = json.load(train_logs_file)

                            train_results = train_logs['train']
                            valid_results = train_logs['valid']

                            list_keys = list(train_results[0].keys())

                            train_list = []
                            valid_list = []
                            for key in list_keys:
                                train_list.append([])
                                valid_list.append([])
                            
                            for train_result, valid_result in zip(train_results, valid_results):

                                for i, key in enumerate(list_keys):
                                    train_list[i].append(train_result[key])
                                    valid_list[i].append(valid_result[key])
                            
                            # PLOT RESULTS
                            run_graphics_path = run + '/graphics/'
                            if os.path.isdir(run_graphics_path):
                                os.system('rm -rf {}'.format(run_graphics_path))
                            os.mkdir(run_graphics_path)

                            for key, t_item, v_item in zip(list_keys, train_list, valid_list):
                                plot(key, t_item, v_item, run_graphics_path)

                            #print('Fold size: ' + str(len(train_list)))
                            # GET THE MEAN RESULTS OF RUNS
                            if r == 0:
                                train_list_runs = train_list
                                valid_list_runs = valid_list
                            else:
                                for c in range(len(train_list_runs)):
                                    train_list_runs[c] = [x + y for x,y in zip(train_list_runs[c], train_list[c])]
                                    valid_list_runs[c] = [x + y for x,y in zip(valid_list_runs[c], valid_list[c])]
                        
                        for t in range(len(train_list_runs)):
                            train_list_runs[t] = [x / len(runs) for x in train_list_runs[t]]
                            valid_list_runs[t] = [x / len(runs) for x in valid_list_runs[t]]

                        #print('Mean Fold Size: ' + str(len(train_list_runs)))
                        
                        # PLOT THE MEAN RESULT
                        os.mkdir(graphics_path)
                        for key, t_item, v_item in zip(list_keys, train_list_runs, valid_list_runs):
                                plot(key, t_item, v_item, graphics_path)

                        if e == 0:
                                encoder_train = train_list_runs
                                encoder_valid = valid_list_runs
                        else:
                            for c in range(len(encoder_train)):
                                encoder_train[c] = [x + y for x,y in zip(encoder_train[c], train_list_runs[c])]
                                encoder_valid[c] = [x + y for x,y in zip(encoder_valid[c], valid_list_runs[c])]
                    
                    for t in range(len(encoder_train)):
                        encoder_train[t] = [x / len(encoders) for x in encoder_train[t]]
                        encoder_valid[t] = [x / len(encoders) for x in encoder_valid[t]]

                    decoders_train.append(encoder_train)
                    decoders_valid.append(encoder_valid)

                save_path = dataset_path + 'graphics/'
                if os.path.isdir(save_path):
                    os.system('rm -rf {}'.format(save_path))
                os.mkdir(save_path)

                for k, key in enumerate(list_keys):
                    for dn, (dtrain, dvalid, decoder) in enumerate(zip(decoders_train, decoders_valid, decoders)):
                        train = dtrain[k]
                        valid = dvalid[k]
                        y = list(range(len(train)))

                        #p, e = getminmax(train, valid)
                        p, e = 0, 0
                        title = ''
                        if dataset == 'covid19china/':
                            title = 'CC-CCII'
                            p, e = 30, 80
                        elif dataset == 'covid20cases/':
                            p, e = 60, 100
                            title = 'Zenodo'
                        elif dataset == 'medseg/':
                            title = 'MedSeg'
                            p, e = 10, 70
                        elif dataset == 'mosmed/':
                            title = 'MosMed'
                            p, e = 40, 90
                        elif dataset == 'ricord1a/':
                            title = 'Ricord1a'
                            p, e = 70, 100
                        #print(p, e)

                        if decoder == 'unetplusplus/':
                            dec = 'U-net++'
                        elif decoder == 'unet/':
                            dec = 'U-net'
                        elif decoder == 'fpn/':
                            dec = 'FPN'
                        elif decoder == 'pspnet/':
                            dec = 'PSPNet'
                        elif decoder == 'linknet/':
                            dec = 'LinkNet'
                        elif decoder == 'manet/':
                            dec = 'MA-Net'

                        ylabel = ''
                        if key == 'Fscore':
                            ylabel = 'F-score'
                        else:
                            ylabel = key

                        yticks = [p/100 for p in range(p, e, 5)]
                        xticks = [p for p in range(0, len(train)+5, 5)]
                        #plt.figure(figsize=[10.4, 6.8])
                        #plt.rc('font', size=20)
                        plt.plot(y, train, '-', lw=2, color=colors[dn], label='Train of ' + dec)
                        plt.plot(y, valid, '--', lw=2, color=colors[dn], label='Valid of ' + dec)
                        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                        plt.xlabel('Epochs')
                        plt.ylabel(ylabel)
                        plt.xticks(xticks)
                        plt.yticks(yticks)
                        plt.title(title)
                        plt.tight_layout()
                    plt.grid()
                    plt.savefig(save_path + key.lower().replace(' ','_').replace(':','_') + '.pdf')
                    plt.clf()
                
            '''
                    if e == 0:
                        encoders_results_train = []
                        encoders_results_valid = []
                        for key in list_keys:
                            encoders_results_train.append([])
                            encoders_results_valid.append([])
                    
                    for t, (t_item, v_item) in enumerate(zip(train_list_runs, valid_list_runs)):
                        encoders_results_train[t].append(t_item)
                        encoders_results_valid[t].append(v_item)

                    print('Encoder size:' + str(len(encoders_results_train)))

                decoders_train.append(encoders_results_train)
                decoders_valid.append(encoders_results_valid)
            
            print('Decoder size:' + str(len(decoders_train)))

            for k, key in enumerate(list_keys):
                for dn, (dtrain, dvalid, decoder) in enumerate(zip(decoders_train, decoders_valid, decoders)):
                    train = dtrain[k]
                    valid = dvalid[k]
                    
                    print(len(train))        
                    t = np.transpose(np.array(train))
                    print(t.shape)
                    train = t.cumsum(axis=0)
                    print(train.shape)
                    t_mean = t.mean(axis=1)
                    print(t_mean.shape)
                    t_std = t.std(axis=1)

                    v = np.transpose(np.array(valid))
                    valid = v.cumsum(axis=0)
                    v_mean = v.mean(axis=1)
                    v_std = v.std(axis=1)

                    y = list(range(50))
                    p, e = getminmax(t_mean, v_mean)
                    yticks = [p/100 for p in range(p, e, 5)]
                    xticks = [p for p in range(0, len(train)+5, 5)]
                    plt.plot(y, t_mean, '-', lw=2, label='Train ' + decoder.replace('/',''), color=colors[dn])
                    plt.plot(y, v_mean, '--', lw=2, label='Valid ' + decoder.replace('/',''), color=colors[dn])
                    plt.fill_between(y, t_mean+t_std, t_mean-t_std, facecolor=colors[dn], alpha=0.2)
                    plt.fill_between(y, v_mean+v_std, v_mean-v_std, facecolor=colors[dn], alpha=0.2)
                    #plt.set_title(r'random walkers empirical $\mu$ and $\pm \sigma$ interval')
                    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                    #plt.set_xlabel('num steps')
                    #plt.set_ylabel('position')
                    plt.yticks(yticks)
                    plt.tight_layout()
                    plt.grid()
                plt.savefig('{}.png'.format(list_keys[k]))
                plt.clf()            
            '''
