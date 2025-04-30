import torch
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


def get_weights_file_path(model_folder, model_basename, epoch: str):
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(model_folder, model_basename):
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) 


def get_device():
    """
    Automatically selects 'cuda' if available, 'mps' if available (for Apple silicon),
    otherwise returns 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu" 
    
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))  # ignore this class in evaluation
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious) 


def compute_dice(pred, target, num_classes):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2 * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-8)
        dices.append(dice_score.item())
    return np.mean(dices)

def compute_pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item() 
    

def main():
    # get_device
    device = get_device()
    print("Using device:", device)


if __name__=="__main__":
    main()
