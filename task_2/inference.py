import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import yaml 
import argparse

from model import build_model  
from utils import (
    get_device, 
    latest_weights_file_path,
    get_weights_file_path
)

# Preprocessing for inference; ensure this matches what your model expects.
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
]) 

def inference_on_image(model, image_path, device):
    """
    Run inference on a single image and return the segmentation prediction.
    """
    # Load and preprocess the test image.
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]
    
    with torch.no_grad():
        logits = model(input_tensor)  # [1, num_classes, H, W]
        if logits.shape[1] == 1:
            # For binary segmentation output (with one channel).
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
        else:
            # For multi-class segmentation, select the class with maximum probability.
            probs = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).unsqueeze(1).float()  # shape: [1, 1, H, W]
    
    # Convert prediction to numpy array and remove extra dimensions.
    pred_mask = pred_mask.cpu().numpy().squeeze()
    return image, pred_mask

def save_prediction(image_pil, pred_pil, save_path, gt_pil=None):
    """Accept PIL images directly and save a figure side by side."""
    # Convert PIL to np.array for matplotlib
    image_np = np.array(image_pil)
    pred_np = np.array(pred_pil)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(pred_np, cmap="jet")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if gt_pil:
        gt_np = np.array(gt_pil)
        axes[2].imshow(gt_np, cmap="jet")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    else:
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



def main():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("Read the configuration parameters")
    parser.add_argument('--config_file', required=True, help="Config file for model building and training.")
    args = parser.parse_args()

    # config file
    with open(args.config_file, "r") as f: 
        config = yaml.safe_load(f)
    
    # Get device once.
    device = get_device()
    
    # Retrieve directories from the configuration.
    # Ensure your config file includes these keys under an appropriate section (e.g. "inference")
    test_images_dir = config["inference"]["test_images_dir"]
    inference_results_dir = config["inference"]["inference_results_dir"]

    # Build the model.
    in_channels = config["model"]["in_channels"] 
    out_channels = config["model"]["out_channels"]
    model = build_model(
        in_channels=in_channels, 
        out_channels=out_channels
    ).to(device)
        
    # Model loading.
    model_folder = config["model"]["model_folder"] 
    model_basename = config["model"]["model_basename"] 
    preload = config["model"]["preload"]

    if os.path.exists(model_folder):
        print(f"The path {model_folder} exists.")
    else:
        print(f"The path {model_folder} does not exist.")
        
    model_filename = None
    if preload == 'latest':
        model_filename = latest_weights_file_path(model_folder, model_basename)
    else: 
        model_filename = get_weights_file_path(model_folder, model_basename, preload) if preload else None
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        print('No model to preload, starting from scratch')
    
    # Ensure output directory exists.
    os.makedirs(inference_results_dir, exist_ok=True)
    
    # Run inference on each image in the test directory.
    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        image, pred_mask = inference_on_image(model, img_path, device)
        
        # Save the predicted mask as an image (scaling mask values to [0, 255]).
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        output_path = os.path.join(inference_results_dir, f"pred_{img_name}")


        save_prediction(image, mask_img, output_path, gt_pil=None)
        print(f"Saved prediction for {img_name} to {output_path}")
  

if __name__ == "__main__":
    main()
