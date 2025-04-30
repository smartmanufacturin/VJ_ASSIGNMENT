import torch 
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import yaml 
import os 
import numpy as np 
from utils import (
    get_device, 
    get_weights_file_path,
    latest_weights_file_path,
    compute_dice, 
    compute_iou, 
    compute_pixel_accuracy,
)


from loss import DiceCrossEntropyLoss, SimpleLoss, WeightedCrossEntropyLoss
from model import build_model
from coco_datasets import CocoSegmentationDataset, compute_sample_weights   # datasets 
from PIL import Image  # Needed for transforms
import argparse

 


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0 
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # [B, num_classes, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0) 
        
        progress_bar.set_postfix(train_loss=loss.item())
        
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device, num_classes, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []
    pixel_accuracies = [] 
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes)
            dice = compute_dice(preds, masks, num_classes)
            acc = compute_pixel_accuracy(preds, masks)
            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(acc)
            
            progress_bar.set_postfix(val_loss=loss.item())
    
    val_loss = running_loss / len(dataloader.dataset) 
    return val_loss, np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies) 



import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def save_prediction(image, gt, pred, epoch, predictions_dir):
    """Save a plot of the image, ground truth, and prediction side by side.

    Args:
        image (torch.Tensor): Input image tensor [C, H, W].
        gt (torch.Tensor): Ground truth mask [H, W].
        pred (torch.Tensor): Predicted mask [H, W].
        epoch (int): Current epoch.
        predictions_dir (str): Directory to save the plot.
    """
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="jet")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="jet")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(predictions_dir, exist_ok=True)
    file_path = os.path.join(predictions_dir, f"sample_epoch_{epoch}.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)




def main():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("Read the configuration parameters")
    parser.add_argument('--config_file', required=True, help="Config file for model building and training.")
    args = parser.parse_args()

    # config file
    with open(args.config_file, "r") as f: 
        config = yaml.safe_load(f)

    # device
    device = get_device()

    # prediction dir for saving the image and mask and predicted mask
    predictions_dir = config["training"]["predictions_dir"]

    # Data
    
    images_train_dir = config["dataset"]["images_train_dir"]
    masks_train_dir = config["dataset"]["masks_train_dir"]  

    images_val_dir = config["dataset"]["images_val_dir"]
    masks_val_dir = config["dataset"]["masks_val_dir"]  
   
    
    
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
    ])
    
    batch_size = config["training"]["batch_size"]
    train_dataset = CocoSegmentationDataset(images_train_dir, masks_train_dir, image_transform, mask_transform) 
    val_dataset = CocoSegmentationDataset(images_val_dir, masks_val_dir, image_transform, mask_transform)  

    sample_weights, class_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=8, shuffle=False)




    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Model parameters
    in_channels = config["model"]["in_channels"] 
    out_channels = config["model"]["out_channels"]
    model = build_model(
        in_channels=in_channels, 
        out_channels=out_channels
    ).to(device) 

    # Training settings
    # criterion_train = DiceCrossEntropyLoss() 
    criterion_train = WeightedCrossEntropyLoss(class_weights=class_weights) 
    criterion_train = criterion_train.to(device)

    criterion_val = SimpleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = config["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    experiment_name = config["training"]["experiment_name"]
    writer = SummaryWriter(log_dir=experiment_name) 
    
    # model preloading 
    model_folder = config["model"]["model_folder"] 
    model_basename = config["model"]["model_basename"] 
    preload = config["model"]["preload"]  

    if os.path.exists(model_folder):
        print(f"The path {model_folder} exists.")
    else:
        print(f"The path {model_folder} does not exist.")
        os.makedirs(model_folder)
        print(f"The path {model_folder} created.")

    initial_epoch = 0
    global_step = 0
    model_filename = None

    if preload == 'latest':
        model_filename = latest_weights_file_path(model_folder, model_basename) 
    else: 
        model_filename = get_weights_file_path(model_folder, model_basename, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # start training
    for epoch in range(initial_epoch, num_epochs + initial_epoch):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_train, device, epoch, num_epochs + initial_epoch-1)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, iou, dice, pixel_acc = validate_epoch(
                                                        model,
                                                        val_loader,
                                                        criterion_val,
                                                        device,
                                                        out_channels,  
                                                        epoch,
                                                        num_epochs + initial_epoch - 1
                                                    ) 

        print(f"Val Loss: {val_loss:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {pixel_acc:.4f}")
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/IoU", iou, epoch)
        writer.add_scalar("Metrics/Dice", dice, epoch)
        writer.add_scalar("Metrics/Pixel_Acc", pixel_acc, epoch) 
        
        if (epoch+1) % 10 == 0 or epoch==0: 

            sample_idx = np.random.randint(0, len(val_dataset))
            image_sample, mask_sample = val_dataset[sample_idx] 
            image_tensor = image_sample.unsqueeze(0).to(device)
    
            with torch.no_grad():
                output = model(image_tensor)  # output shape: [1, num_classes, H, W]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()  # shape: [H, W]
            
            save_prediction(image_sample, mask_sample, pred, epoch, predictions_dir=predictions_dir)



            model_filename = get_weights_file_path(
                model_folder=model_folder, 
                model_basename=model_basename,
                epoch=f"{epoch:02d}"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename) 

        scheduler.step() 
        torch.cuda.empty_cache()

    writer.close() 
    
if __name__ == "__main__": 
    main()
