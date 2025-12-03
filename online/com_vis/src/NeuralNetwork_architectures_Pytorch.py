# -----------------------------------------------------------------------------
# Copyright (c) 2025 Andrea Dal Prete - Politecnico di Milano
# All rights reserved.
#
# This script is part of the research published in:
# [Your Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Andrea Dal Prete (andrea.dalprete@polimi.it)
# -----------------------------------------------------------------------------

# This script contains the neural network architecture constructor functions. This means that in the main 
# script the functions contained in this script can be called to built the Neural Network model architecture
# and use it for inference/training. The model is construct in Pytorch.  

# import libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F


# this is the constructor customized dinov2 model. In this script we take the DINOv2 model, we cut the output layer, and 
# we stuck an additional attention block and multilayer perception to solve the payload classification problem.

def get_customizedDINOv2(dino_model, n):
    class CustomizedDINOv2(nn.Module): # define a class 
        def __init__(self, dino_model, num_classes=n):
            super(CustomizedDINOv2, self).__init__()

            # Load dinov2 model from facebook research
            self.dinov2_model = torch.hub.load('facebookresearch/dinov2', dino_model)

            # Freeze all the layers in dinov2 model
            for param in self.dinov2_model.parameters():
                param.requires_grad = False 

            # get the dimensions of dinov2 output embeddings
            if dino_model == 'dinov2_vits14':
                self.embedding_dim = 384
            elif dino_model == 'dinov2_vitb14':
                self.embedding_dim = 768
            elif dino_model == 'dinov2_vitl14':
                self.embedding_dim = 1024
            else:
                self.embedding_dim = 1536

            # define a single head self attention layer, for more information see "Attention Is All You Need", link: https://arxiv.org/abs/1706.03762
            self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=1, dropout=0.4, batch_first=True)
            # p.s.: important to specify "batch_first=True" if your data have the following dimensions: (num_batches, target 
            # sequence length, embedding_dimension)

            self.classifier = nn.Sequential( # define the multilayer perception layer for the final output classification
                nn.Linear(self.embedding_dim, 256), 
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes), 
            )

        # define the forward pass function 
        def forward(self, x):
            # output of dinov2
            features = self.dinov2_model.forward_features(x)

            # extract the tensor of features 
            features_tensor = features['x_norm_patchtokens']

            # single head self-attention 
            attn_output, _ = self.attention(features_tensor, features_tensor, features_tensor)

            # final multilayer perception classifier 
            output = self.classifier(attn_output.mean(dim=1))

            # apply final softmax 
            output = F.softmax(output, dim=1)

            return output

    return CustomizedDINOv2(dino_model, n)