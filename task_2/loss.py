import torch 
from torch import nn  
import torch.nn.functional as F
from typing import Dict


class SimpleLoss(nn.Module):
    
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.loss_a = None
        self.loss_p = None
    
    def forward(self, logits, target):
        loss = F.cross_entropy(logits, target)
        return loss


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, dice_smooth=1e-6):
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth

    def forward(self, logits, targets):
        # logits: [B, num_classes, H, W]
        # targets: [B, H, W] or one-hot encoded with shape [B, num_classes, H, W]
        num_classes = logits.shape[1]
        
        # Compute Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # Apply softmax over the channel dimension
        probs = torch.softmax(logits, dim=1)
        
        # If targets are not one-hot, convert to one-hot:
        if targets.dim() == 3:
            targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            targets_onehot = targets.float()
        
        dice_loss = 0.0
        for c in range(num_classes):
            prob_c = probs[:, c, :, :]
            target_c = targets_onehot[:, c, :, :]
            intersection = torch.sum(prob_c * target_c)
            dice_c = (2 * intersection + self.dice_smooth) / (torch.sum(prob_c) + torch.sum(target_c) + self.dice_smooth)
            dice_loss += (1 - dice_c)
        dice_loss = dice_loss / num_classes

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss
 


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for multi-class segmentation."""
    
    def __init__(self, class_weights: Dict[int, float]):
        super().__init__()
        sorted_keys = sorted(class_weights.keys())
        weight_list = [class_weights[k] for k in sorted_keys]
        self.register_buffer("weights", torch.tensor(weight_list, dtype=torch.float))
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy."""
        return self.ce_loss(logits, targets)