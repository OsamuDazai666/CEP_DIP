import os
import torch
from torchvision import datasets, transforms
from torch import optim

from architecture.effnet import EffNet
from helpers.train import train_model

# Create dataloaders
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
main_folder = "dataset"
train_folder = os.path.join(main_folder, "train")
test_folder = os.path.join(main_folder, "test")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

# Create datasets
train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

# Check class-to-index mapping
print("Class to Index Mapping:", train_dataset.class_to_idx)


model = EffNet()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

train_model(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer=optimizer,
    criterion=loss_fn,
    batch_size=32, 
    output_folder="EffNet_Weights",
    epochs=60,  
)
