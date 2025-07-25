# How to use
This repository contains the code to obtain the results presented in the report of the Practical Work. Here you can find the implementation of the EDSR network, as well as the necessary code to train it and showcase it. In the following sections, the individual scripts will be explained. Furthermore the necessary environment needs to be installed with the accompanying ``env.yml`` file. To do so, simply run `conda env create -f env.yml`. Please note that the used cuda version can differ, depending of the target system. The dataset can be downloaded [here](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images). The models used to create the showcased images are contained in a link to the Google Drive and will be shared upon request. Also sometimes the `lpips` module needs to be installed manually with `pip install lpips`.

## How to train the network
To train the network, the train_edsr.py script is needed. This script enables the training of the EDSR network, either from scratch or with a pretrained file, with either the plain L1 loss or a combination of L1 and FSL loss presented in the report. Simply call it like this: `python train_edsr.py`. The script also features various command line arguments, which are listed below:  
- `--dataset_path`: The root path to the dataset on which the model should be trained. Also, the directory must contain the dataset split intop training and validation sets with the name "train" and "valid".
- `--scale`: The upscaling factor the model should be trained on. please note that it is beneficial to train models of x3 and x4 scales on the back of x2 scale models 
- `--epochs`: Number of epochs the model should be trained
- `--batch_size`: The size of the minibatch used during training
- `--lr`: The learning rate used during training
- `--save_path`: The path under which the model should be saved
- `--num_filters`: The maximum number of filters for the convolutions
- `--num_blocks`: The number of residual blocks used
- `--device`: The device the training should take place on
- `--patience`: The number of epoch of which the training should be stopped if no decrease in validation loss is detected
- `--pre_train`: Whether or not a previous trained model should be used to instantiate a higher scale model
- `--frequency_loss`: Whether or not to use the combined loss discussd in the report

Please note that even though default values are provided in the script, the path to the dataset needs to be specified. To use the pre training strategy in the paper, the pre_train flag needs to be explicitly set. The same applies when using the combined loss (`--frequency_loss`).

## How to showcase the model
To generate outputs and to showcase the model, the `showcase.py` script must be used. Simply call it like this: `python showcase.py` with the additional arguments discussed below:  
- `--model_path`: The location under which the to be used model is saved
- `--device`: The device the processing should take place on
- `--img_path`: The path to the to be upscaled image. The image needs to be in a folder called `imgs` and in a subfolder called `ground_truth`. Otherwise it needs to be changed in the `main` function. Additionally, there needs to be the folder `low_res` and `upscaled` present in the same directory as the `ground_truth` folder.
- `--scale`: The scale of the upscaling
- `--num_filters`: The learning rate used during training
- `--num_filters`: The maximum number of filters for the convolutions
- `--num_blocks`: The number of residual blocks used
- `--dataset_path`: The root path to the dataset on which the model should be trained
- `--comp_name`: The name of the comparison plot
- `--out_name`: The name of the whole upscaled image
- `--lr_name`:The names of the corresponding low resolution images
- `--patch_vis`: The name of the patch-wise comparison plots  

It should be noted that the `showcase.py` script outputs various images, namely:  
- `comparison plots`: Plots the low resolution, the ground truth and the super resolved image side by side with various evaluation metrics
- `output`: A single super resolved image
- `low resolution images`: The low resolution version of the image
- `patches`: A side by side comparison of the low resolution, super resolved and ground truth patches    

The overall workflow is that the image is split into smaller chunks which get super resolved individually. Then those super resolved image patches get stitched back together. Please note that while default values are provided, the location of the saved model needs to be specified. For the to-be-upscaled image, follow the specifications discussed above.
