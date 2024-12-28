from architecture.effnet import EffNet
from architecture.lenet5 import LeNet5
from architecture.tinyvgg import TinyVgg

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import os

dataset = ImageFolder("dataset/train")
classes = dataset.classes

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(model, folder_path, transformer, classes, output_file):
    output_df = pd.DataFrame(columns=["Truth", "Prediction", "ImagePath"])
    model.eval()
    for cls in os.listdir(folder_path):
        class_pth = os.path.join(folder_path, cls)
        for img_name in os.listdir(class_pth):
            print(f"Running Model For: {img_name}")
            img_pth = os.path.join(class_pth, img_name)
            img = Image.fromarray(cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB))
            tensor_img = transformer(img).unsqueeze(0)

            with torch.no_grad():
                pred_idx = torch.argmax(model(tensor_img), )
                prediction = classes[pred_idx]
                print(f"pred:{prediction}")

                output_df = pd.concat([output_df, pd.DataFrame({
                    "Truth": [cls], 
                    "Prediction": [prediction], 
                    "ImagePath": [img_pth],
                })])
    
            output_df.to_csv(os.path.join("predictions", output_file))
            


effnet_model = EffNet()
effnet_model.load_state_dict(torch.load("weights/EffNet_Weights/best_model.pth", weights_only=True, map_location=device))

lenet_model = LeNet5()
lenet_model.load_state_dict(torch.load("weights/LeNet5_Weights/best_model.pth", weights_only=True, map_location=device))

tiny_model = TinyVgg(in_channels=3, inter_channels=32, classes=4)
tiny_model.load_state_dict(torch.load("weights/TinyVgg_Weights/best_model.pth", weights_only=True, map_location=device))


transform = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224
    T.ToTensor(),         # Convert images to tensors
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

models = [effnet_model, lenet_model, tiny_model]
output_file = ["effnet_pred.csv", "lenet_pred.csv", "tiny_pred.csv"]

for model, out_file_name in zip(models, output_file):
    predict(
        model=model,
        folder_path="dataset/test",
        transformer=transform,
        classes=classes,
        output_file=out_file_name,
    )