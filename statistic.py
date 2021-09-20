import json
import glob
from pprint import pprint
from scipy import stats

def print_list(my_list):
    size = len(my_list)
    list_str = '['
    for i in range(2):
        list_str += str(my_list[i]) + ', '
    list_str += str(my_list[2]) + ' ... '
    for i in range(2):
        list_str += str(my_list[size - 3 + i]) + ','
    list_str += str(my_list[size-1]) + ']'
    print(list_str)

datasets = ['covid19china', 'covid20cases',  'medseg', 'mosmed', 'ricord1a']

decoders = ['fpn', 'linknet', 'manet', 'pspnet', 'unet', 'unetplusplus']

encoders = ['densenet121', 'resnet101', 'resnext50_32x4d', 'se_resnext101_32x4d', 'timm-regnetx_004', 
            'timm-regnety_004', 'timm-res2net50_26w_4s', 'densenet169', 'resnet50', 'se_resnet101',
            'se_resnext50_32x4d', 'timm-regnetx_006', 'timm-regnety_006', 'vgg16', 'densenet201',
            'resnext101_32x8d',  'se_resnet50', 'timm-regnetx_002', 'timm-regnety_002', 'timm-res2net101_26w_4s']

RUNS = 'RUNS/baseline/'

final_result = []
for decoder in decoders:
    print(decoder)
    decoder_list = []
    decoder_result = []

    for encoder in encoders:
        print(encoder)
        encoder_result = []

        for dataset in datasets:
            #print(dataset)
            
            runs_path = RUNS + dataset + '/' + decoder + '/' + encoder
            runs = glob.glob(runs_path + '/*')
            results_runs = []
            for run in runs:
                if 'graphics' not in run:
                    individual_logs_path = run + '/individual_logs_last.json'
                    with open(individual_logs_path) as f:
                        data = json.load(f)
                        data = data['fscores']
                        #data = [round(i, 3) for i in data]
                        #print(data)
                        if len(data) == 0:
                            print('ERRO ERRO ERRO ERRO ERRO ERRO!!!!!!')
                    results_runs.append(data)
            dataset_result = [sum(x) / len(x) for x in zip(*results_runs)]
            dataset_result = [round(i, 2) for i in dataset_result]

            #print('Mean for dataset:')
            #print_list(dataset_result)

            encoder_result.extend(dataset_result)
        
        #print('Encoder result:')
        #print_list(encoder_result)
        #print('Size: ' + str(len(encoder_result)))

        shapiro_test = stats.shapiro(encoder_result)
        #print(shapiro_test)
        #print(shapiro_test, format(shapiro_test.pvalue, ".60f"))
        #print('\n\n')
        
        decoder_list.append(encoder_result)
        #print('Decoder list: ' + str(len(decoder_list)))
    
    #### Fazer comparação de encoders para decoder
    friedman = stats.f_oneway(decoder_list[0],decoder_list[1],decoder_list[2],decoder_list[3],decoder_list[4],
                                       decoder_list[5], decoder_list[6],decoder_list[7],decoder_list[8],decoder_list[9],
                                       decoder_list[10], decoder_list[11],decoder_list[12],decoder_list[13],decoder_list[14],
                                       decoder_list[15], decoder_list[16],decoder_list[17],decoder_list[18],decoder_list[19])
    #print(decoder_list[0])
    #print(decoder_list[1])
    #print(decoder_list[2])
    #print(decoder_list[3])
    #friedman = stats.f_oneway(decoder_list[0],decoder_list[1],decoder_list[2],decoder_list[3],decoder_list[4])

    print('----------------------------------------------')
    print('Decoder:' + decoder)
    print('Encoder comparison: ')                                   
    print(friedman, format(friedman.pvalue, ".60f"))
    print('----------------------------------------------')
    print('\n\n')

    #----------------
    for item in decoder_list:
        decoder_result.extend(item)
    final_result.append(decoder_result)
    #print('Final result: ' + str(len(final_result)))

#### Fazer comparação de decoders
friedman = stats.f_oneway(final_result[0],final_result[1],final_result[2],final_result[3],final_result[4], final_result[5])

print('----------------------------------------------')
print('Decoder comparison: ')
print(friedman)
print('----------------------------------------------')
print('\n\n')

            
