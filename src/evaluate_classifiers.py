import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm

from classifier_dataset import get_dataloaders
from classifier_models import UNetClassifier, get_resnet18, get_densenet121

def evaluate_model(model, test_loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"Evaluating {model_name} on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability for class 1 (Pneumonia)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return all_labels, all_preds, all_probs

def main(data_dir):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    _, _, test_loader = get_dataloaders(data_dir, batch_size=32, target_size=(224, 224))
    
    models_to_test = {
        'unet': UNetClassifier(num_classes=2),
        'resnet': get_resnet18(num_classes=2, pretrained=False),
        'densenet': get_densenet121(num_classes=2, pretrained=False)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        weights_path = f"best_{name}_classifier.pth"
        if not os.path.exists(weights_path):
            print(f"Weights for {name} not found at {weights_path}. Skipping.")
            continue
            
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, name)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        
    if not results:
        print("No models were evaluated.")
        return
        
    out_dir = "classification_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metrics to text file
    txt_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"{'Model':<15} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}\n")
        f.write("-" * 60 + "\n")
        for name, res in results.items():
            f.write(f"{name.upper():<15} | {res['accuracy']:.4f}   | {res['precision']:.4f}    | {res['recall']:.4f}   | {res['f1']:.4f}\n")
        f.write("="*60 + "\n")
    print(f"Saved metrics summary to {txt_path}")
        
    # Print metrics to console
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name.upper():<15} | {res['accuracy']:.4f}   | {res['precision']:.4f}    | {res['recall']:.4f}   | {res['f1']:.4f}")
    print("="*60 + "\n")
    
    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
        
    class_names = ["Normal", "Pneumonia"]
    for i, (name, res) in enumerate(results.items()):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Confusion Matrix: {name.upper()}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")
    
    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        if 'fpr' in res:
            plt.plot(res['fpr'], res['tpr'], label=f"{name.upper()} (AUC = {res['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classifiers")
    parser.add_argument("--data_dir", type=str, default=r"D:\Semester 7\biomedical image provessing\data\train", help="Path to data")
    
    args = parser.parse_args()
    main(args.data_dir)
