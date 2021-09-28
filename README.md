# Installation: 

pip install -r requirements.txt

copy the files inside the scripts to the utils folders inside the lib/python3.6/site-packages/segmentation_models_pytorch folder

# Dataset

setup a txt file with the images paths

path/to/image1.jpg path/to/mask1.png <br>
path/to/image2.jpg path/to/mask2.png <br>
path/to/image3.jpg path/to/mask3.png

setup the config file following the examples in the config folder

# Training

python main.py --configs configs/confs_train.yml

# Eval

python main.py --configs configs/confs_eval.yml
