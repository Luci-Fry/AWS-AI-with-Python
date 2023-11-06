"""
This file is used to predict the name of flowers that are given as input to it. 

It takes a checkpoint, rebuilds the model using the checkpoint and predicts flower names using input images. 


"""
import torch
from torch import nn
from utility_functions import *
import json 

import numpy as np


#Get the input arguments 
input_args = get_args()

########################################################################

# load the model's checkpoint and rebuild the model using the load_checkpoint function
model = load_checkpoint(input_args.checkpoint_path)

#########################################################################

#If the user wants to use the GPU, check to see if the GPU is available 
#if the GPU is not available, tell the user and use CPU

if input_args.gpu:
    if torch.cuda.is_available():
        print('GPU is available... Using GPU')
        interface = torch.device('cuda')
        
    else: 
        print('GPU is not available... Turn on if possible... Using CPU')
        interface = torch.device('cpu')

#if the option has not been given, just predict on the CPU
else:
    interface = torch.device('cpu')
#print(input_args.gpu)

#print('All alright')

########################################################################

#retrieve the top k most likely probabilities 
topk = input_args.top_k

#migrate the model to the appropriate interface 
model.to(interface)

#set the model into evaluation mode 
model.eval()

#process the image with the process_image function
image = process_image(input_args.image_path)

#change the image form a numpy array to a torch tensor
image = torch.from_numpy(image)
    
"""unsqueeze adds the batch size to the size of the image, the model expects the size to have four dimensions """

image = image.unsqueeze_(0)
    
    
''' image.float() below 
    converts the input to a FloatTensor instead of a DoubleTensor 
    to match the model because the weights and biases of the model are 
    by default FloatTensors'''

image = image.float()    
    
#migrate the image to the appropriate interface 
image = image.to(interface)  
    
with torch.no_grad():

    top_probs, top_classes = [],[]
    output_logps = model.forward(image)
    
    ps = torch.exp(output_logps)
    
    top_p, top_index = ps.topk(topk, dim = 1)
    
#     #sort the indices from largest to smallest 
#     top_index_sorted = sort([top_index[0][i].item() for i in range(len(top_index))])
    
    
    #make a mapping from the output index to the classes 
    idx_to_class = {indx: class_ for class_,indx in model.class_to_idx.items()}
    
    #make a list out of the tensor for the top_p probabilities 
    for i in range(topk):
        top_probs.append(top_p[0][i].item())
        
    #map from index to class
    for i in range(topk):
        idx = top_index[0][i].item()
        clas = idx_to_class[idx]
        top_classes.append(clas)
    
    #Get the matching from category to name
    with open(input_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    
    #create a list of the class names from the indices and classes 
    class_names = []
    for indx in top_classes:
        class_names.append(cat_to_name[indx])
        
    print("Name of Flower:  Probability")    
    for i in range(len(class_names)):
        
        #print('\n')
        print(f"{class_names[i]}:   {top_probs[i]}")
        #print('\n')
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        