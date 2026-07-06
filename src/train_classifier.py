import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from classifier_dataset import get_dataloaders
from classifier_models import UNetClassifier, get_resnet18, get_densenet121

def train(model_name, data_dir, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    if model_name == 'unet':
        model = UNetClassifier(num_classes=2)
    elif model_name == 'resnet':
        model = get_resnet18(num_classes=2, pretrained=True)
    elif model_name == 'densenet':
        model = get_densenet121(num_classes=2, pretrained=True)
    else:
        raise ValueError("Invalid model name. Choose from 'unet', 'resnet', 'densenet'")
        
    model = model.to(device)
    
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size, target_size=(224, 224))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    save_path = f"best_{model_name}_classifier.pth"
    
    # History for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"--- Training {model_name.upper()} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} (Val Acc: {best_val_acc:.2f}%)")
            
    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plotting Learning Curves
    out_dir = "classification_results"
    os.makedirs(out_dir, exist_ok=True)
    
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name.upper()} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title(f'{model_name.upper()} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_learning_curves.png"))
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pneumonia Classifier")
    parser.add_argument("--model", type=str, required=True, choices=['unet', 'resnet', 'densenet'], help="Model architecture")
    parser.add_argument("--data_dir", type=str, default=r"D:\Semester 7\biomedical image provessing\data\train", help="Path to data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train(args.model, args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
