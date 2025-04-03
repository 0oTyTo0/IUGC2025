import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from heatmap_dataset import HeatmapLandmarkDataset
from heatmap_net import get_heatmap_model, HeatmapLoss, extract_coordinates, euclidean_distance

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_coord_distance = 0.0
    
    with tqdm(train_loader, desc=f"Training Epoch {epoch}") as pbar:
        for i, (images, heatmaps, landmarks) in enumerate(pbar):
            # Move data to GPU
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            landmarks = landmarks.to(device)
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Calculate heatmap loss
            loss = criterion(pred_heatmaps, heatmaps)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Extract coordinates from heatmaps (for evaluation only, not for gradient computation)
            with torch.no_grad():
                pred_coords = extract_coordinates(pred_heatmaps)
                
                # Calculate coordinate distance (for monitoring training effect)
                coord_distance = torch.mean(euclidean_distance(pred_coords, landmarks))
            
            # Update total loss and distance
            batch_loss = loss.item()
            total_loss += batch_loss
            total_coord_distance += coord_distance.item()
            
            # Update progress bar
            pbar.set_postfix(loss=f"{batch_loss:.4f}", coord_dist=f"{coord_distance.item():.4f}")
            
            # Add to TensorBoard
            iteration = (epoch - 1) * len(train_loader) + i
            writer.add_scalar('Loss/train_batch', batch_loss, iteration)
            writer.add_scalar('Coord_Distance/train_batch', coord_distance.item(), iteration)
    
    avg_loss = total_loss / len(train_loader)
    avg_coord_distance = total_coord_distance / len(train_loader)
    
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.add_scalar('Coord_Distance/train_epoch', avg_coord_distance, epoch)
    
    return avg_loss, avg_coord_distance

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory with timestamp
    if args.timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = os.path.join(args.save_dir, f"unet_heatmap_{timestamp}")
    elif args.model_suffix:
        args.save_dir = os.path.join(args.save_dir, f"unet_heatmap_{args.model_suffix}")
    else:
        args.save_dir = os.path.join(args.save_dir, "unet_heatmap")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    
    # Save current configuration parameters
    with open(os.path.join(args.save_dir, 'training_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TensorBoard logger
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # Basic data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    train_dataset = HeatmapLandmarkDataset(
        csv_file=args.train_csv,
        img_dir=args.train_dir,
        transform=train_transform,
        train=True,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = get_heatmap_model(num_keypoints=args.num_keypoints, heatmap_size=args.heatmap_size)
    model = model.to(device)
    
    # Print model structure
    print(model)
    
    # Define loss function
    criterion = HeatmapLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Load pretrained weights if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            lowest_train_loss = checkpoint.get('train_loss', float('inf'))
            best_train_coord_distance = checkpoint.get('train_coord_distance', float('inf'))
            print(f"Starting from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            start_epoch = 1
            lowest_train_loss = float('inf')
            best_train_coord_distance = float('inf')
    else:
        start_epoch = 1
        lowest_train_loss = float('inf')
        best_train_coord_distance = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        # Train one epoch
        train_loss, train_coord_distance = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
        )
        
        # Learning rate adjustment (based on fixed intervals)
        scheduler.step()
        
        # Output training information
        print(f"Epoch {epoch}/{args.epochs} - Training Loss: {train_loss:.4f}, Coordinate Distance: {train_coord_distance:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model at specified intervals
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_coord_distance': train_coord_distance
            }, checkpoint_path)
            print(f"Saved model to {checkpoint_path}")
        
        # Save model with lowest training loss
        if train_loss < lowest_train_loss:
            lowest_train_loss = train_loss
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', 'best_train_loss_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': lowest_train_loss,
                'train_coord_distance': train_coord_distance
            }, checkpoint_path)
            print(f"Saved best training loss model to {checkpoint_path}")
        
        # Save model with lowest coordinate distance
        if train_coord_distance < best_train_coord_distance:
            best_train_coord_distance = train_coord_distance
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', 'best_train_coord_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_coord_distance': best_train_coord_distance
            }, checkpoint_path)
            print(f"Saved best coordinate distance model to {checkpoint_path}")
        
        # Save final model
        if epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', 'final_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_coord_distance': train_coord_distance
            }, checkpoint_path)
            print(f"Saved final model to {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Image Landmark Heatmap Regression Training Script (Training Set Only)")
    
    # Dataset parameters
    parser.add_argument('--train_csv', type=str, default='train/label.csv',
                        help='Path to training CSV file')
    parser.add_argument('--train_dir', type=str, default='train/labelled',
                        help='Directory with training images')
    
    # Heatmap parameters
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Size of the heatmap')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Standard deviation for Gaussian heatmap')
    parser.add_argument('--num_keypoints', type=int, default=3,
                        help='Number of keypoints')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lr_step', type=int, default=15,
                        help='Learning rate decay step (decay every n epochs)')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate decay factor')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='results_heatmap_train_only',
                        help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Interval to save model (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--timestamp', action='store_true',
                        help='Add timestamp to save directory to avoid overwriting previous results')
    parser.add_argument('--model_suffix', type=str, default='',
                        help='Model name suffix to distinguish different training configurations')
    
    args = parser.parse_args()
    
    main(args) 