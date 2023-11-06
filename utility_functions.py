""" utility_functions includes fucntions called in the predict.py to help predict the closs of new flower images using the train.py script. It includes fucntions to:
~ load in the checkpoint 
~ process the input image 
~ show the image tensor 
~ 

"""
import torch
from torch import nn
from PIL import Image
from torchvision import models 
from collections import OrderedDict
import argparse

import numpy as np

###########################################################################

def get_args():
    
    """This function gets input arguments of various forms from the user in the command line. Allows user to define:
    1. image_path: path to image to predict on
    2. checkpoint_path: path to checkpoint
    3. --top_k: number of top k most likely classes to return 
    4. --category_names: name of json file that has the mapping from categories to real names
    5. --gpu : indicate wheater GPU should be used for inference
    
    """
    parser = argparse.ArgumentParser(description = 'Gets arguments like the image directory, checkpoint directory, etc from the user', prefix_chars = '-+/')
    parser.add_argument('image_path',help= 'path to data directory')
    parser.add_argument('checkpoint_path', help = 'path to checkpoint', type= str)
    parser.add_argument('--top_k', default = '1', type = int, help = 'number of top k most likely classes to return')
    parser.add_argument('--category_names', default = 'cat_to_name.json', type= str, help ='name of json file that has the mapping from categories to real names' )
    parser.add_argument('//gpu', '--gpu', action = 'store_true', help = 'indicate wheater GPU should be used for training')
    
    return parser.parse_args()
    
    
###########################################################################

def load_checkpoint(filepath):
    
    """ 
    This function loads in a given checkpoint and rebuilds the model
    
    Arguments:
    - filepath : path to the checkpoint 
    
    Returns:
    The rebuilt model 
    
    """
    input_args = get_args()
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if 'vgg19' in input_args.checkpoint_path:
        model = models.vgg19(pretrained=True)
        model_name = 'vgg19'
        
    elif 'densenet121' in input_args.checkpoint_path:
        model = models.densenet121(pretrained=True) 
        model_name = 'densenet121'
        
    #get the number of input features 
    in_features = {'vgg19':25088, 'densenet121':1024}
    
    myclassifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(in_features[model_name], 2000)),
                                          ('relu', nn.ReLU()),
                                          ('Drop1', nn.Dropout(p= 0.2)),
                                          ('fc2',nn.Linear(2000, 500)),
                                          ('Drop2', nn.Dropout(p= 0.1)),
                                         ('relu', nn.ReLU()),
                                          ('fc3',nn.Linear(500, 102)),
                                         ('output', nn.LogSoftmax(dim= 1)) 
                                         ]))
    model.classifier = myclassifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = checkpoint['cat_to_name_mapping']
    
    return model

##########################################################################

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    
    Arguments:
    - image: path to image 
    
    Returns:
    - Numpy array of image 
    '''
    
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    
    
    # open the image using PIL
    im = Image.open(image)
 

    # resize the image with the thumbnail function
    im.thumbnail((256, 256))
        
    #crop the image with crop method 
    width, height = im.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
        
    im = im.crop((left, top, right, bottom))
    

    #change the image to a numpy array    
    im = np.array(im)/255   
    
    #Normalize with the means and standard deviation 
    
    im = (im - means) / std

    #transpose with ndarray.transpose so that the color channel will be the first dimension in the PIL Image 
    im = im.transpose()
    
        
    return im


###########################################################################

def imshow(image, ax=None, title=None, name= None):
    
    """ 
    Imshow for Tensor.
    
    Arguments:
    - image: image tensor
    
    Returns:
    Show the input image tensor as an actual image 
    
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
   
    
    ax.imshow(image)
    ax.set_title(name)
    
    return ax

######################################################################################







