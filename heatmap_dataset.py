import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import numpy as np

class HeatmapLandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, heatmap_size=64, sigma=2.0):
        """
        Ultrasound Image Landmark Heatmap Dataset
        
        Args:
            csv_file (str): Path to the CSV file with landmark coordinates
            img_dir (str): Directory with images
            transform (callable, optional): Transform to be applied on images
            train (bool): Whether this is training set
            heatmap_size (int): Size of the heatmap
            sigma (float): Standard deviation for Gaussian heatmap
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Image preprocessing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def generate_heatmap(self, center_x, center_y, height, width):
        """Generate a Gaussian heatmap for a single keypoint"""
        x = np.arange(0, width, 1, np.float32)
        y = np.arange(0, height, 1, np.float32)
        y = y[:, np.newaxis]
        
        # Calculate heatmap center
        x0 = center_x
        y0 = center_y
        
        # Generate Gaussian distribution heatmap
        heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def __getitem__(self, index):
        # Get image filename
        row = self.data.iloc[index]
        filename = row['Filename']
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply image transform
        image = self.transform(image)
        
        # Extract landmark coordinates and normalize
        ps1 = ast.literal_eval(row["PS1"])
        ps2 = ast.literal_eval(row["PS2"])
        aop = ast.literal_eval(row["FH1"]) 
        
        # Original image size (assuming 512x512)
        img_width, img_height = 512, 512
        
        # Create heatmap labels - one channel per keypoint, 3 channels total
        heatmaps = np.zeros((3, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # Scale coordinates to heatmap size
        scale_x = self.heatmap_size / img_width
        scale_y = self.heatmap_size / img_height
        
        # Generate heatmap for each keypoint
        keypoints = [ps1, ps2, aop]
        for i, kp in enumerate(keypoints):
            # Convert coordinates to heatmap size
            x = int(kp[0] * scale_x)
            y = int(kp[1] * scale_y)
            
            # Ensure coordinates are within heatmap range
            x = max(0, min(x, self.heatmap_size - 1))
            y = max(0, min(y, self.heatmap_size - 1))
            
            # Generate heatmap
            heatmaps[i] = self.generate_heatmap(x, y, self.heatmap_size, self.heatmap_size)
        
        # Convert to tensor
        heatmaps = torch.from_numpy(heatmaps)
        
        # Save original coordinates (for evaluation metrics)
        landmarks = [
            ps1[0] / img_width, ps1[1] / img_height,
            ps2[0] / img_width, ps2[1] / img_height,
            aop[0] / img_width, aop[1] / img_height
        ]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return image, heatmaps, landmarks 
