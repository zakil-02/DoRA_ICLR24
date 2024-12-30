# test the model checkpoint.pth 

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from vision_transformer import vit_small
import utils
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import weights pth file
pretrained_weights = 'checkpoint.pth'
model = vit_small()
utils.load_pretrained_weights(model, pretrained_weights, "teacher", "vit_small", 16)
output_dir = './'
os.makedirs(output_dir, exist_ok=True)


# Define preprocessing pipeline
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),#, scale=(0.4, 1.), interpolation=Image.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Load an image
image = Image.open("image.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Extract features
with torch.no_grad():
    features = model(input_tensor)

    print("#features :", len(features))
    print("features[0].shape :", features[0].shape)
    print("features[1].shape :", features[1].shape)
    print("features[2].shape :", features[2].shape)
    print("features[3].shape :", features[3].shape)
    print("features[4].shape :", features[4].shape)

    # # Save the attention map
    # fig = plt.figure(figsize=(16, 8))
    # attention_map = features[2][0, 1].detach().cpu().numpy()  # [197, 197]
    # print("attention_map.shape :", attention_map.shape)
    # plt.imshow(attention_map, cmap='hot', interpolation='nearest', alpha=0.5)
    # plt.colorbar()
    # plt.show()
    # plt.savefig(os.path.join(output_dir, "attention_map.png"))
    # # Plot all the masks on the original image with different colors
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # ax.imshow(image)
    
    # for i in range(1, features[2].shape[1]):  # Skip the class token
    #     attention_map = features[2][0, i].detach().cpu().numpy()
    #     attention_map = cv2.resize(attention_map, (image.width, image.height))
    #     attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    #     ax.imshow(attention_map, alpha=0.1)
    
    # plt.axis('off')
    # plt.show()
    # plt.savefig(os.path.join(output_dir, "all_attention_maps.png"))

    # Create a color palette for patches
    num_patches = features[2].shape[1]  # Number of patches (excluding the class token)
    colors = plt.colormaps.get_cmap('tab20').colors  # Generate distinct colors

    # Plot the original image with masks
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(image)
    
    # Initialize the output image
    output_image = np.array(image, dtype=np.float32)

    for i in range(1, num_patches):  # Skip the class token
        attention_map = features[2][0, i].detach().cpu().numpy()  # Extract attention map for patch `i`

        # Resize to match image dimensions
        attention_map = cv2.resize(attention_map, (image.width, image.height))
        
        # Normalize the attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Threshold the attention map
        threshold = 0.3  # Highlight only pixels with attention above this value
        attention_highlight = np.where(attention_map > threshold, attention_map, 0)
        
        # Create a colored mask using the patch color
        color = np.array(colors[i])[:3]  # Get RGB values (0-1)
        mask = np.zeros_like(image, dtype=np.float32)
        for c in range(3):  # Apply color to each channel
            mask[..., c] = attention_highlight * color[c] * 255  # Scale to RGB range

        # Blend only where attention is non-zero
        non_zero_mask = attention_highlight > 0
        for c in range(3):  # Apply the mask color only to relevant areas
            output_image[..., c] = np.where(non_zero_mask, 
                                            (0.6 * mask[..., c] + 0.4 * output_image[..., c]),  # Blend
                                            output_image[..., c])

    # Save and visualize the result
    output_image = output_image.astype(np.uint8)
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "all_attention_maps_clean.png"), bbox_inches='tight')
    plt.show()