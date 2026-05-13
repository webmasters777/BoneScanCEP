#!/usr/bin/env python3
"""
ResNet50 Training Script for Bone Metastasis Detection using PyTorch.
Trains a fine-tuned ResNet50 model on the bone scan dataset.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

# Dataset path
DATASET_PATH = "dataset project"

def load_dataset_labels(view_type="RANT"):
    """Load labels from the dataset text file."""
    labels_file = os.path.join(DATASET_PATH, f"chest{view_type}", f"chest{view_type}.txt")
    labels_dict = {}
    
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                filename, label = parts
                labels_dict[filename] = int(label)
    
    return labels_dict

class BoneScanDataset(Dataset):
    """Custom dataset for bone scan images."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_resnet50_model():
    """Create and modify ResNet50 model."""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

def train_resnet50():
    """Train ResNet50 model."""
    print("Loading dataset...")
    
    # Get all image paths and labels
    image_dir = os.path.join(DATASET_PATH, "chestRANT")
    labels_dict = load_dataset_labels(view_type="RANT")
    
    image_paths = []
    labels = []
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if filename not in labels_dict:
            continue
        image_paths.append(os.path.join(image_dir, filename))
        labels.append(labels_dict[filename])
    
    print(f"Dataset loaded: {len(image_paths)} samples")
    print(f"Normal: {np.sum(np.array(labels)==0)}, Metastasis: {np.sum(np.array(labels)==1)}")
    
    # Split dataset
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train_paths)}, Test: {len(X_test_paths)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BoneScanDataset(X_train_paths, y_train, transform=train_transform)
    test_dataset = BoneScanDataset(X_test_paths, y_test, transform=test_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_resnet50_model().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 20
    best_accuracy = 0.0
    
    print("Training ResNet50...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels_batch.float())
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'bone_scan_resnet50_final.pth')
    
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    # Final evaluation
    model.load_state_dict(torch.load('bone_scan_resnet50_final.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(labels_batch.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Metastasis']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return model

if __name__ == "__main__":
    train_resnet50()