import os
import glob
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def copy(images, masks, datapath):
    for img, msk in zip(images, masks):
        nimg = img.replace('/all/',datapath)
        nmsk = msk.replace('/all/',datapath)

        os.system('cp {} {}'.format(img, nimg))
        os.system('cp {} {}'.format(msk, nmsk))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required='true')
    parser.add_argument('--size', type=float, required='true')
    args = parser.parse_args()

    os.mkdir('datasets/' + args.dataset + '/train')
    os.mkdir('datasets/' + args.dataset + '/train/images')
    os.mkdir('datasets/' + args.dataset + '/train/masks')

    os.mkdir('datasets/' + args.dataset + '/valid')
    os.mkdir('datasets/' + args.dataset + '/valid/images')
    os.mkdir('datasets/' + args.dataset + '/valid/masks')

    ipath = 'datasets/' + args.dataset + '/all/images/'
        
    masks = []
    images = glob.glob(ipath + '*.jpg')
    for image in images:
        masks.append(image.replace('/images/','/masks/').replace('.jpg','.png'))
    
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=args.size)

    print(len(images))
    print(len(X_train))
    print(len(X_test))

    copy(X_train, y_train, '/train/')
    copy(X_test, y_test, '/valid/')