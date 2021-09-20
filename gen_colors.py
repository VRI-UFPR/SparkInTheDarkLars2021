import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, required='true')
    args = parser.parse_args()

    with open('colors.txt','w') as colors_file:
        for i in range(args.num_classes):
            if i == 0:
                color = '0, 0, 0'
            else:
                b = random.randint(50, 255)
                g = random.randint(50, 255)
                r = random.randint(50, 255)  
                color = '{}, {}, {}'.format(b, g, r)
            if i < (args.num_classes - 1):
                color += '\n'
            colors_file.write(color)