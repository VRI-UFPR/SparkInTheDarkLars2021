import os
import cv2
import numpy as np

datasets = ['covid19china', 'covid20cases', 'medseg', 'mosmed', 'ricord1a']

datasets_path = 'datasets/'

for dataset in datasets:
    dataset_path = datasets_path + dataset + '/'

    sets = ['train', 'valid']
    for s in sets:
        spath = dataset_path + s + '/'
        ids_path = spath + 'ids.txt'

        images_path = spath + 'images/'
        masks_path = spath + 'masks/'

        empty_images_path = spath + 'empty_images/'
        empty_masks_path = spath + 'empty_masks/'

        if os.path.isdir(empty_images_path):
            os.system('rm -r ' + empty_images_path)
        os.mkdir(empty_images_path)

        if os.path.isdir(empty_masks_path):
            os.system('rm -r ' + empty_masks_path)
        os.mkdir(empty_masks_path)

        with open(ids_path) as ids_file:
            lines = ids_file.readlines()

        for line in lines:
            #print(line)
            id = line.replace('\n','')

            img_path = images_path + id + '.jpg'
            msk_path = masks_path + id + '.png'

            #print(msk_path)
            mask = cv2.imread(msk_path, 0)

            uniques = np.unique(mask)

            if len(uniques) == 1:
                new_img_path = empty_images_path + id + '.jpg'
                new_msk_path = empty_masks_path + id + '.png'

                mv_img = 'mv ' + img_path + ' ' + new_img_path
                mv_msk = 'mv ' + msk_path + ' ' + new_msk_path

                print(mv_img)
                print(mv_msk)

                os.system(mv_img)
                os.system(mv_msk)