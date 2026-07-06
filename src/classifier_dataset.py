import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Define classes (0: Normal, 1: Pneumonia)
        self.classes = {"NORMAL": 0, "PNEUMONIA": 1}
        
        for class_name, label in self.classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            paths = glob.glob(os.path.join(class_dir, "*.*"))
            paths = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            if max_samples:
                paths = paths[:max_samples]
            
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Convert all to RGB so standard pretrained models (like ResNet/DenseNet) work without modification
        # and so they have 3 channels.
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        
        # If the underlying dataset returned the image directly (which it does, since we removed transform there)
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, batch_size=32, target_size=(224, 224)):
    # Standard ImageNet normalization since we might use pretrained weights
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation/Test should not have random augmentations
    eval_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = PneumoniaDataset(data_dir, transform=None, max_samples=250) # Limited for fast CPU demo
    
    # Deterministic split: 70% Train, 15% Val, 15% Test
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_ds = TransformSubset(train_subset, train_transform)
    val_ds = TransformSubset(val_subset, eval_transform)
    test_ds = TransformSubset(test_subset, eval_transform)
    
    # num_workers=0 is safer on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
