import csv
import glob
import json

run = '0p1_100elr4/noda'
datasets = ['covid19china','covid20cases','mosmed','medseg','ricord1a']

csv_file = open('da_results.csv','w')

writer = csv.writer(csv_file)

writer.writerow([' '] + datasets)

row = ['Sem Data Augmentation']
for dataset in datasets:
    runs_path = 'RUNS/' + run + '/' + dataset + '/unetplusplus/timm-regnetx_002/'
    runs_list = glob.glob(runs_path + '*')

    fscore = 0
    for r in runs_list:
        json_result = r + '/train_logs.json'
        with open(json_result) as json_file:
            result = json.load(json_file)
        fscore += result['valid'][99]['Fscore'] / 5
    row.append(round(fscore,3))
writer.writerow(row)

das = ['Clahe/', 'CoarseDropout/', 'ElasticTransform/', 'Emboss/', 'Flip/', 
       'GaussianBlur/', 'GridDistortion/', 'GridDropout/', 'ImageCompression/', 'MedianBlur/',
       'OpticalDistortion/', 'PiecewiseAffine/', 'Posterize/', 'RandomBrightnessContrast/', 'RandomCrop/',
       'RandomGamma/', 'RandomSnow/', 'Rotate/']#, 'Sharpen/', 'ShiftScaleRotate/']
#das = ['Clahe/', 'CoarseDropout/', 'ElasticTransform/', 'Emboss/', 'Flip/',
#        'GaussianBlur/', 'GridDistortion/', 'MedianBlur/', 'OpticalDistortion/', 'ShiftScaleRotate/']
super_runs = ['0p1_100elr4/']

for da in das:
    for super_run in super_runs:
        if '0p1' in super_run:
            row = [da.replace('/','') + ' P=0.1']
        else:
            row = [da.replace('/','') + ' P=0.2']

        for dataset in datasets:
            runs_path = 'RUNS/' + super_run + da + dataset + '/unetplusplus/timm-regnetx_002/'
            runs_list = glob.glob(runs_path + '*')

            fscore = 0
            for r in runs_list:
                json_result = r + '/train_logs.json'
                with open(json_result) as json_file:
                    result = json.load(json_file)
                fscore += result['valid'][99]['Fscore'] / 5
            row.append(round(fscore,3))       
        writer.writerow(row)
        
csv_file.close
