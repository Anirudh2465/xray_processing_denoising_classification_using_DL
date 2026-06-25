import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class DenoisingDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.noisy_dir = os.path.join(dataset_dir, "noisy")
        self.clean_dir = os.path.join(dataset_dir, "clean")
        
        # Get all noisy images
        self.noisy_paths = glob.glob(os.path.join(self.noisy_dir, "*.png"))
        
    def __len__(self):
        return len(self.noisy_paths)
        
    def __getitem__(self, idx):
        noisy_path = self.noisy_paths[idx]
        filename = os.path.basename(noisy_path)
        base_name = filename.split("_noise_")[0]
        clean_path = os.path.join(self.clean_dir, base_name + ".png")
        
        noisy_img = Image.open(noisy_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
            
        return noisy_img, clean_img

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.upconv3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return self.sigmoid(out)

def train_model(dataset_dir, epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = DenoisingDataset(dataset_dir, transform=transform)
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item() * noisy_imgs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    torch.save(model.state_dict(), "unet_denoiser_model.pth")
    print("Model saved to unet_denoiser_model.pth")
    
    return model, val_loader

def evaluate_and_plot(model, dataloader, num_images=5):
    device = torch.device("cpu")
    model.eval()
    
    noisy_imgs, clean_imgs = next(iter(dataloader))
    noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
    
    with torch.no_grad():
        outputs = model(noisy_imgs)
        
    # Plotting
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 4*num_images))
    for i in range(num_images):
        # Noisy
        ax = axes[i, 0]
        ax.imshow(noisy_imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Noisy Input")
        ax.axis('off')
        
        # Denoised
        ax = axes[i, 1]
        ax.imshow(outputs[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Denoised Output")
        ax.axis('off')
        
        # Clean
        ax = axes[i, 2]
        ax.imshow(clean_imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Clean Target")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("denoising_results.png")
    print("Saved evaluation results to denoising_results.png")

if __name__ == "__main__":
    DATASET_DIR = r"D:\Semester 7\biomedical image provessing\poisson_dataset"
    
    # Train the model
    model, val_loader = train_model(DATASET_DIR, epochs=10, batch_size=32)
    
    # Evaluate and save plots
    evaluate_and_plot(model, val_loader)
