import cv2
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required='true')
    parser.add_argument('--num_classes', type=int, required='true')
    args = parser.parse_args()

    dataset_ids = 'datasets/{}/train/ids.txt'.format(args.dataset)

    with open(dataset_ids, 'r') as txt_file:
        ids = txt_file.read().splitlines()

    heights = []
    widths = []
    proportions = [0]*args.num_classes
    masks_path = 'datasets/{}/train/masks/'.format(args.dataset)
    for idx in ids:
        mask = cv2.imread(masks_path + idx + '.png', 0)
        height, width = mask.shape
        #heights.append(height)
        #widths.append(width)

        size = height * width

        masks = np.zeros((height, width, args.num_classes))
        for i, unique_value in enumerate(np.unique(mask)):
            masks[:, :,unique_value][mask == unique_value] = 1
        
        masks = masks.transpose(2, 0, 1)

        for i, mask in zip(range(args.num_classes), masks):
            #print(i)
            r = np.count_nonzero(mask)
            #print(r)
            p = r / size
            #print(p)
            #print('--------')
            proportions[i] += p
    
    #print(proportions)
    #print(len(ids))
    #props = [i/len(ids) for i in proportions]
    #print(props)

    norm = [float(i)/sum(proportions) for i in proportions]
    print(norm)    

    #mean_height = sum(heights) / len(heights)
    #mean_width = sum(widths) / len(widths)

    #print(mean_height, mean_width)