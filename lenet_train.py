import os
import torch
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.optim import Adam

from architecture.lenet5 import LeNet5
from helpers.train import train_model

device = "gpu" if torch.cuda.is_available() else "cpu"

model = LeNet5() # lenet-5 base model

optim = Adam(model.parameters(), lr=0.0001)
loss = CrossEntropyLoss()

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

train_model(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer=optim,
    criterion=loss,
    epochs=60,
    batch_size=32,
    output_folder="Lenet_Weights",
    device=device
)