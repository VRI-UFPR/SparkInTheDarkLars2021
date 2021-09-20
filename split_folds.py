import argparse
import numpy as np
from sklearn.model_selection import KFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required='true')
    parser.add_argument('--num_folds', type=int, required='true')
    args = parser.parse_args()

    ######################################## Train and Valid ########################################
    
    dataset_ids = 'datasets/{}/train/ids.txt'.format(args.dataset)

    with open(dataset_ids, 'r') as txt_file:
        ids = txt_file.read().splitlines()
    
    ids_all = 'datasets/{}/train/ids_all.txt'.format(args.dataset)
    with open(ids_all, 'w') as txt_file:
        for i, idx in zip(range(len(ids)), ids):
            image_path = 'datasets/{}/train/images/{}.jpg'.format(args.dataset, idx)
            mask_path = 'datasets/{}/train/masks/{}.png'.format(args.dataset, idx)

            if i == len(ids) - 1:
                txt_file.write('{} {}'.format(image_path, mask_path))    
            else:
                txt_file.write('{} {}\n'.format(image_path, mask_path)) 

    ids = np.array(ids)
    
    kf = KFold(n_splits=args.num_folds, shuffle=True)
    
    cont = 0
    for train, valid in kf.split(ids):
        train_path = 'datasets/{}/train/train_ids{}.txt'.format(args.dataset, cont)
        valid_path = 'datasets/{}/train/valid_ids{}.txt'.format(args.dataset, cont)
        cont += 1

        with open(train_path, 'w') as train_file:
            for i, idx in zip(range(len(train)), train):
                image_path = 'datasets/{}/train/images/{}.jpg'.format(args.dataset, ids[idx])
                mask_path = 'datasets/{}/train/masks/{}.png'.format(args.dataset, ids[idx])
                if i == len(train) - 1:
                    train_file.write('{} {}'.format(image_path, mask_path))    
                else:
                    train_file.write('{} {}\n'.format(image_path, mask_path))

        with open(valid_path, 'w') as valid_file:
            for i, idx in zip(range(len(valid)), valid):
                image_path = 'datasets/{}/train/images/{}.jpg'.format(args.dataset, ids[idx])
                mask_path = 'datasets/{}/train/masks/{}.png'.format(args.dataset, ids[idx])
                if i == len(valid) - 1:
                    valid_file.write('{} {}'.format(image_path, mask_path))    
                else:
                    valid_file.write('{} {}\n'.format(image_path, mask_path))
    
    ######################################## Test ########################################
    
    dataset_ids = 'datasets/{}/valid/ids.txt'.format(args.dataset)
    
    with open(dataset_ids, 'r') as txt_file:
        ids = np.array(txt_file.read().splitlines())

    test_path = 'datasets/{}/valid/test_ids.txt'.format(args.dataset)

    with open(test_path, 'w') as test_file:
        for i in range(len(ids)):
            image_path = 'datasets/{}/valid/images/{}.jpg'.format(args.dataset, ids[i])
            mask_path = 'datasets/{}/valid/masks/{}.png'.format(args.dataset, ids[i])
            if i == len(ids) - 1:
                test_file.write('{} {}'.format(image_path, mask_path))    
            else:
                test_file.write('{} {}\n'.format(image_path, mask_path))