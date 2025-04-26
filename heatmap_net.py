import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapUNet(nn.Module):
    """UNet-based Heatmap Regression Network"""
    def __init__(self, num_keypoints=3, heatmap_size=64):
        super(HeatmapUNet, self).__init__()
        
        # Save heatmap size as attribute
        self.heatmap_size = heatmap_size
        
        # Encoder part (downsampling path)
        # Input: [batch_size, 3, 512, 512]
        self.enc_conv1 = self._double_conv(3, 64)     # -> [batch_size, 64, 512, 512]
        self.pool1 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 64, 256, 256]
        
        self.enc_conv2 = self._double_conv(64, 128)   # -> [batch_size, 128, 256, 256]
        self.pool2 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 128, 128, 128]
        
        self.enc_conv3 = self._double_conv(128, 256)  # -> [batch_size, 256, 128, 128]
        self.pool3 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 256, 64, 64]
        
        self.enc_conv4 = self._double_conv(256, 512)  # -> [batch_size, 512, 64, 64]
        self.pool4 = nn.MaxPool2d(kernel_size=2)      # -> [batch_size, 512, 32, 32]
        
        # Bottleneck part
        self.bottleneck = self._double_conv(512, 1024) # -> [batch_size, 1024, 32, 32]
        
        # Decoder part (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # -> [batch_size, 512, 64, 64]
        self.dec_conv4 = self._double_conv(1024, 512)  # 512 + 512 = 1024 (skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # -> [batch_size, 256, 128, 128]
        self.dec_conv3 = self._double_conv(512, 256)   # 256 + 256 = 512 (skip connection)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # -> [batch_size, 128, 256, 256]
        self.dec_conv2 = self._double_conv(256, 128)   # 128 + 128 = 256 (skip connection)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # -> [batch_size, 64, 512, 512]
        self.dec_conv1 = self._double_conv(128, 64)    # 64 + 64 = 128 (skip connection)
        
        # Final output layer - generate one heatmap channel for each keypoint
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1) # -> [batch_size, num_keypoints, 512, 512]
                                                                       # Finally resized to [batch_size, num_keypoints, 64, 64]
    
    def _double_conv(self, in_channels, out_channels):
        """Two consecutive convolutional layers, each followed by batch normalization and ReLU activation"""
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
        
        # Final layer output heatmaps
        heatmaps = self.final_layer(d1)  # [batch_size, num_keypoints, 512, 512]
        
        # Ensure output size matches target heatmap size
        if heatmaps.size(2) != self.heatmap_size or heatmaps.size(3) != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=(self.heatmap_size, self.heatmap_size), 
                                     mode='bilinear', align_corners=True)  # -> [batch_size, num_keypoints, 64, 64]
        
        return heatmaps

def get_heatmap_model(num_keypoints=3, heatmap_size=64):
    """
    Get heatmap regression model
    
    Args:
        num_keypoints (int): Number of keypoints
        heatmap_size (int): Size of the heatmap
    
    Returns:
        nn.Module: Heatmap regression model
    """
    return HeatmapUNet(num_keypoints=num_keypoints, heatmap_size=heatmap_size)

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
    
    # Find the maximum response location for each heatmap
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    # Convert to 2D coordinates
    y_coords = torch.div(max_indices, width, rounding_mode='floor').float() / height
    x_coords = (max_indices % width).float() / width
    
    # Merge coordinates
    coords = torch.zeros(batch_size, num_keypoints * 2, device=heatmaps.device)
    for i in range(num_keypoints):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    
    return coords

# Heatmap loss function
class HeatmapLoss(nn.Module):
    """Heatmap MSE Loss Function"""
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Calculate MSE loss between predicted and target heatmaps
        
        Args:
            pred_heatmaps: Predicted heatmaps [batch_size, num_keypoints, height, width]
            target_heatmaps: Target heatmaps [batch_size, num_keypoints, height, width]
        """
        return self.criterion(pred_heatmaps, target_heatmaps)

# Distance calculation function
def euclidean_distance(pred, target):
    """Calculate Euclidean distance between predicted and target points"""
    pred = pred.view(-1, 3, 2)  # Reshape to (batch_size, 3, 2), each point has x,y coordinates
    target = target.view(-1, 3, 2)
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=2))  # Calculate Euclidean distance for each keypoint 
