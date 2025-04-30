import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from typing import Optional, Callable, Tuple, Dict

class CocoSegmentationDataset(Dataset):
    """
    Dataset for COCO segmentation images and corresponding masks.
    Returns a tuple (image, mask) for each sample.
    If no label mapping is provided, it is computed automatically by extracting unique pixel values from the masks.
    An inverse mapping is also computed to convert sequential labels back to original labels.
    """
    def __init__(self, 
                 images_dir: str, 
                 masks_dir: str, 
                 transform_image: Optional[Callable] = None, 
                 transform_mask: Optional[Callable] = None,
                 label_mapping: Optional[Dict[int, int]] = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir 
        self.masks_files = sorted(os.listdir(masks_dir))  # list of mask filenames

        self.transform_image = transform_image or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.transform_mask = transform_mask or transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
        ])
        
        # Compute or assign the label mapping.
        if label_mapping is None:
            self.label_mapping = self._compute_label_mapping()
            print("Computed label mapping:", self.label_mapping)
        else:
            self.label_mapping = label_mapping
            
        # Compute inverse mapping to convert sequential labels back to original labels.
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def __len__(self) -> int:
        return len(self.masks_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.masks_files):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.masks_files)}.")

        # Load the mask and image.
        mask_filename = self.masks_files[idx]
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask = Image.open(mask_path)

        image_path = None
        for ext in [".jpg", ".png"]:
            temp_path = os.path.join(self.images_dir, mask_filename.split(".")[0] + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break  # Stop once we find a valid image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations.
        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        # Remap mask labels using the label mapping.
        remapped_mask = torch.zeros_like(mask)
        for orig_label, new_label in self.label_mapping.items():
            remapped_mask[mask == orig_label] = new_label
        mask = remapped_mask
        
        return image, mask

    def _compute_label_mapping(self) -> Dict[int, int]:
        """
        Scans all mask images to compute a mapping from original pixel values 
        to new sequential labels.
        Returns:
            A dictionary where keys are original pixel values and values 
            are new sequential labels starting from 0.
        """
        unique_labels = set()
        for mask_filename in self.masks_files:
            mask_path = os.path.join(self.masks_dir, mask_filename)
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file '{mask_filename}' does not exist; skipping.")
                continue
            try:
                mask_img = Image.open(mask_path)
                mask_array = np.array(mask_img)
                unique_labels.update(np.unique(mask_array).tolist())
            except Exception as e:
                print(f"Warning: Could not process mask '{mask_filename}': {e}")
                continue
        sorted_labels = sorted(unique_labels)
        return {orig: new for new, orig in enumerate(sorted_labels)}
    
    def decode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a mask with sequential labels back to original label values.
        Args:
            mask (torch.Tensor): A mask tensor with sequential labels.
        Returns:
            torch.Tensor: A mask tensor where each pixel has been mapped back to its original label.
        """
        decoded_mask = torch.zeros_like(mask)
        for seq_label, orig_label in self.inverse_label_mapping.items():
            decoded_mask[mask == seq_label] = orig_label
        return decoded_mask

def compute_sample_weights(dataset: CocoSegmentationDataset) -> torch.Tensor:
    """
    Computes a weight for each sample based on the inverse frequency of the classes present in its mask.
    
    Returns:
        sample_weights: A tensor of weights for each sample in the dataset.
    """
    # First, count how many pixels in total belong to each class across the entire dataset.
    class_pixel_counts = {cls: 0 for cls in dataset.label_mapping.values()}
    total_pixels = 0
    # Looping through the dataset to accumulate counts
    for idx in range(len(dataset)):
        # We only need the mask here
        _, mask = dataset[idx]
        unique_classes, counts = torch.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            class_pixel_counts[cls] += count
            total_pixels += count

    # Compute frequency and then the inverse frequency (weight) for each class.
    # Adding a small epsilon to avoid division by zero.
    epsilon = 1e-6
    class_weights = {}
    for cls, count in class_pixel_counts.items():
        freq = count / total_pixels
        class_weights[cls] = 1.0 / (freq + epsilon)

    # Now, assign a weight to each sample. Here we take the average inverse frequency 
    # for the classes present in the sample’s mask.
    sample_weights = []
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        unique_classes = torch.unique(mask).tolist()
        weight = sum([class_weights[c] for c in unique_classes]) / len(unique_classes)
        sample_weights.append(weight)

    # Normalize weights (optional)
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sample_weights = sample_weights / sample_weights.sum()
    return sample_weights, class_weights 


# test script
def main(): 
    images_dir = "/home/farhad/vjt/task_1/coco_dataset/train2017"
    masks_dir = "/home/farhad/vjt/task_1/coco_dataset/masks_train"
    
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
    ])
    
    # Create the dataset (the label mapping is computed automatically).
    dataset = CocoSegmentationDataset(images_dir, masks_dir, image_transform, mask_transform)
    
    # Compute sample weights for underrepresented classes.
    sample_weights, class_weights = compute_sample_weights(dataset)
    print("Sample weights computed for each sample.")

    # Create a WeightedRandomSampler.
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    
    # Create the DataLoader with the sampler.
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}:")
        print("  Image batch shape:", images.shape)
        print("  Mask batch shape:", masks.shape)
        # Process only one batch 
        break

    image, mask = dataset[0]
    decoded_mask = dataset.decode_mask(mask)
    print("Decoded mask shape:", decoded_mask.shape)

    import matplotlib.pyplot as plt

    # class_weights = {class_index: weight_value, ...}
    class_indices = sorted(class_weights.keys())
    weight_values = [class_weights[idx] for idx in class_indices]
    frequency_values = [1.0 / w if w != 0 else 0 for w in weight_values] 

    plt.bar(class_indices, frequency_values)
    plt.xlabel("Class Index")
    plt.ylabel("Frequency (1 / Weight)")
    plt.title("Per-Class Frequencies")
    plt.grid(True)
    plt.xticks(class_indices, class_indices)
    plt.savefig("freq.png")
    plt.show() 
    

    plt.bar(class_indices, weight_values)
    plt.xlabel("Class Index")
    plt.ylabel("Weight")
    plt.title("Per-Class Weight")
    plt.grid(True)
    plt.xticks(class_indices, class_indices)
    plt.savefig("weights.png")
    plt.show()



    # import matplotlib.pyplot as plt
    # plt.hist(sample_weights.numpy(), bins=30)
    # plt.title("Distribution of Sample Weights")
    # plt.xlabel("Weight")
    # plt.ylabel("Number of Samples")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()