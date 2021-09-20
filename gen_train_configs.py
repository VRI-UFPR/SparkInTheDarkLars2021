import os
import yaml

width = 256
height = 256
num_folds = 5
batch_size = 8
num_epochs = 100
num_workers = 4
learning_rate = 0.0001

mode = 'train'
experiment = 'ShiftScaleRotate'

encoders = ['timm-regnetx_002']

#nodes = ['vti2-ib', 'vti1-ib', 'pti']
nodes = ['vti2-ib', 'vti2-ib', 'vti2-ib']
#nodes = ['vti1-ib', 'vti1-ib', 'vti1-ib']
#nodes = ['pti', 'pti', 'pti']
#nodes = ['vti2-ib']
#decoders = ['unetplusplus', 'unet','fpn','pspnet','linknet', 'pan', 'manet', 'deeplabv3', 'deeplabv3plus']
decoders = ['unetplusplus']#, 'unet','fpn','pspnet','linknet', 'manet']
datasets = ['ricord1a', 'covid20cases', 'mosmed', 'medseg', 'covid19china']
#datasets = ['medseg']
augmentations = ["shift_scale_rotate"]
aug_name = "shift_scale_rotate"
#augmentations = [""]
augmentation_prob = 0.2

print(aug_name)

gpu = 0
node_num = 0
node_count = 0
node_usage = 2

if node_usage * len(nodes) < len(datasets):
    print('Little node usage!')
    exit()

for dataset in datasets:

    sh_cmds = []
    py_cmds = []

    if node_count >= node_usage:
        node_num += 1
        node_count = 0
    
    node_count += 1
    #print(nodes[node_num])

    sh = '#!/bin/sh\n#SBATCH -t 7-00:00:00\n#SBATCH -c 4\n#SBATCH -o /home/bakrinski/segtool/logs/{}_{}_{}_log.out\n\
#SBATCH --job-name={}_{}_{}\n#SBATCH -n 1 #NUM_DE_PROCESSOS\n#SBATCH -p 7d\n#SBATCH -N 1 #NUM_NODOS_NECESSARIOS\n\
#SBATCH --nodelist={}\n#SBATCH --gres=gpu:2\n#SBATCH -e /home/bakrinski/segtool/logs/{}_{}_{}_error.out\n\n\
export PATH="/home/bakrinski/anaconda3/bin:$PATH"\n\nmodule load libraries/cuda/10.1\n\n'.format(dataset, encoders[0], aug_name, 
                                                                                                 dataset, encoders[0], aug_name, 
                                                                                                 nodes[node_num], 
                                                                                                 dataset, encoders[0], aug_name)

    for decoder in decoders:
        for encoder in encoders:    
            for fold in range(num_folds):
                configs = {
                    "general": {"mode": mode, 
                                "num_workers": num_workers,
                                "experiment": experiment, 
                                "dataset": dataset,
                                "gpu": gpu},
                    
                    "model": {"encoder": encoder, 
                              "decoder": decoder, 
                              "batch_size": batch_size,
                              "num_epochs": num_epochs, 
                              "learning_rate": learning_rate,
                              "height": height, 
                              "width": width},

                    "dataset": {"train": "datasets/{}/train/train_ids{}.txt".format(dataset, fold),
                                "valid": "datasets/{}/train/valid_ids{}.txt".format(dataset, fold),
                                "labels": "datasets/{}/labels.txt".format(dataset)},

                    "augmentation": {
                            "augmentations": augmentations,
                            "augmentation_prob": augmentation_prob
                    } 
                }
                configs_name = 'configs/train_' + dataset + '_' + decoder + '_' + encoder + '_fold' + str(fold) + '_' + aug_name + '.yml'
                with open(configs_name, 'w') as config_file:
                    yaml.dump(configs, config_file)

                py_cmds.append("python main.py --configs {}".format(configs_name))
                sh_cmds.append("srun python main.py --configs {}".format(configs_name))

    #gpu += 1
    #if gpu == 2:
    #    gpu = 0

    for sh_cmd in sh_cmds:
        sh += sh_cmd + '\n'

    sh_file = 'train_' + dataset + '_' + encoders[0] + '_' + aug_name + '.sh'
    with open(sh_file,'w') as shf:
        shf.write(sh)
    
    py = 'import os\n\nls=['
    for py_cmd in py_cmds:
        py += '"' + py_cmd + '",\n'
    py += ']\n\nfor l in ls:\n  os.system(l)'

    py_file = 'train_' + dataset + '_' + encoders[0] + '_' + aug_name + '.py'
    with open(py_file,'w') as pyf:
        pyf.write(py)

'''
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
'''