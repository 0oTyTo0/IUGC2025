import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import torch
import os
from torchvision import transforms
from PIL import Image

class HeatmapUNet(nn.Module):
    """UNet-based heatmap regression network"""
    def __init__(self, num_keypoints=3, heatmap_size=64):
        super(HeatmapUNet, self).__init__()
        
        # Save the heatmap size as an attribute
        self.heatmap_size = heatmap_size
        
        # Encoder part (downsampling path)
        # Input: [batch_size, 3, H, W]
        self.enc_conv1 = self._double_conv(3, 64)     # -> [batch_size, 64, H, W]
        self.pool1 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 64, H/2, W/2]
        
        self.enc_conv2 = self._double_conv(64, 128)   # -> [batch_size, 128, H/2, W/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 128, H/4, W/4]
        
        self.enc_conv3 = self._double_conv(128, 256)  # -> [batch_size, 256, H/4, W/4]
        self.pool3 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 256, H/8, W/8]
        
        self.enc_conv4 = self._double_conv(256, 512)  # -> [batch_size, 512, H/8, W/8]
        self.pool4 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 512, H/16, W/16]
        
        # Bottleneck part
        self.bottleneck = self._double_conv(512, 1024) # -> [batch_size, 1024, H/16, W/16]
        
        # Decoder part (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # -> [batch_size, 512, H/8, W/8]
        self.dec_conv4 = self._double_conv(1024, 512)  # 512 + 512 = 1024 (skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # -> [batch_size, 256, H/4, W/4]
        self.dec_conv3 = self._double_conv(512, 256)   # 256 + 256 = 512 (skip connection)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # -> [batch_size, 128, H/2, W/2]
        self.dec_conv2 = self._double_conv(256, 128)   # 128 + 128 = 256 (skip connection)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # -> [batch_size, 64, H, W]
        self.dec_conv1 = self._double_conv(128, 64)    # 64 + 64 = 128 (skip connection)
        
        # Final output layer - generates one heatmap channel for each keypoint
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1) # -> [batch_size, num_keypoints, H, W]
    
    def _double_conv(self, in_channels, out_channels):
        """Two consecutive convolution layers, each followed by batch normalization and ReLU activation"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc_conv1(x)       # First encoder features
        p1 = self.pool1(e1)
        
        e2 = self.enc_conv2(p1)      # Second encoder features
        p2 = self.pool2(e2)
        
        e3 = self.enc_conv3(p2)      # Third encoder features
        p3 = self.pool3(e3)
        
        e4 = self.enc_conv4(p3)      # Fourth encoder features
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path (with skip connections)
        d4 = self.upconv4(b)
        d4 = torch.cat([e4, d4], dim=1)  # Skip connection
        d4 = self.dec_conv4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([e3, d3], dim=1)  # Skip connection
        d3 = self.dec_conv3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)  # Skip connection
        d2 = self.dec_conv2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)  # Skip connection
        d1 = self.dec_conv1(d1)
        
        # Final layer outputs heatmaps
        heatmaps = self.final_layer(d1)
        
        # Ensure output dimensions match the target heatmap size
        if heatmaps.size(2) != self.heatmap_size or heatmaps.size(3) != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=(self.heatmap_size, self.heatmap_size), 
                                     mode='bilinear', align_corners=True)
        
        return heatmaps

# Coordinate extraction function
def extract_coordinates(heatmaps, original_img_size=512):
    """
    Extract keypoint coordinates from heatmaps
    
    Args:
        heatmaps (torch.Tensor): Predicted heatmaps [batch_size, num_keypoints, height, width]
        original_img_size (int): Original image size
    
    Returns:
        torch.Tensor: Extracted coordinates [batch_size, num_keypoints*2]
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    # Find the maximum response position in each heatmap
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    # Convert to 2D coordinates
    y_coords = torch.div(max_indices, width, rounding_mode='floor').float() / height
    x_coords = (max_indices % width).float() / width
    
    # Combine coordinates
    coords = torch.zeros(batch_size, num_keypoints * 2, device=heatmaps.device)
    for i in range(num_keypoints):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    
    return coords

class model:
    def __init__(self):
        '''
        Initialize the model
        '''
        self.model = HeatmapUNet(num_keypoints=3, heatmap_size=64).cpu()
    
    def load(self, path="./"):
        '''
        Load model weights
        '''
        # Try multiple possible model filenames
        possible_model_paths = [
            os.path.join(path, "model_weight.pth"),
            os.path.join(path, "model.pth"),
            os.path.join(path, "heatmap_model.pth")
        ]
        
        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    # Check if it's a checkpoint containing multiple components
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # Use the model state dict instead of the entire checkpoint
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        print(f"Successfully loaded model_state_dict from checkpoint")
                    else:
                        # Try to load directly, assuming it's a simple model state dictionary
                        self.model.load_state_dict(checkpoint)
                    return self
                except Exception as e:
                    print(f"Failed to load model file {model_path}: {e}")
                    continue
        
        # If no model file is found, try loading the default file
        default_model_path = os.path.join(path, "unet_heatmap.pth")
        print(f"No model file found, trying to load from default path: {default_model_path}")
        try:
            checkpoint = torch.load(default_model_path, map_location="cpu")
            # Check if it's a checkpoint containing multiple components
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Use the model state dict instead of the entire checkpoint
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Successfully loaded model_state_dict from checkpoint")
            else:
                # Try to load directly, assuming it's a simple model state dictionary
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load model file: {e}")
            print("Please ensure the model file exists and is named 'unet_heatmap.pth', 'model.pth', or 'heatmap_model.pth'")
        
        return self
    
    def predict(self, X):
        '''
        Prediction function, input an image, output coordinates
        
        Args:
            X: PIL.Image object, the input image
 
        
        Returns:
            coords: numpy array of shape (6,), predicted keypoint coordinates
                    Format: [x1, y1, x2, y2, x3, y3] where:
                    - (x1, y1) is the coordinate of the first keypoint
                    - (x2, y2) is the coordinate of the second keypoint
                    - (x3, y3) is the coordinate of the third keypoint
                    The coordinates should be in the pixel space of the original input image.
        '''
        self.model.eval()

        width, height = X.size
        # Apply the same transformations as during training
        tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        image = tf(X).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

        with torch.no_grad():
            # Forward pass to get heatmaps
            heatmaps = self.model(image)  # (1, 3, 64, 64)
            
            # Extract coordinates from heatmaps (1, 6)
            coords = extract_coordinates(heatmaps) #if you use other methods instead of heatmap ,it is unnecessary.

        # Convert to numpy array (6,)
        coords = coords.squeeze(0).detach().numpy()

        # Convert normalized coordinates back to original image size
        # Even indices (0,2,4) are x coordinates, odd indices (1,3,5) are y coordinates
        coords[::2] *= width   # x coordinates
        coords[1::2] *= height  # y coordinates

        return coords
    
    def save(self, path="./"):
        '''
        Save model weights
        '''
        pass

# IMPORTANT: Input and Output Specification
# ----------------------------------------
# Input (X):   
#   - A PIL.Image object (from PIL.Image.open)
#   - Represents an RGB image 
# 
# Output (coords):
#   - A numpy array of shape (6,) containing 3 keypoint coordinates
#   - Format: [x1, y1, x2, y2, x3, y3]
#   - Coordinates must be in the pixel space of the original input image
#   - The coordinates should represent the exact locations of the detected keypoints
