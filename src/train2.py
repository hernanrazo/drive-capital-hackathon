import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torchvision.datasets as datasets

def get_file_paths(directory: str):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

def get_key_by_value(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)


def main() -> None:

    label_map = {
        0:"basketball",
        1: "car",
        2: "dog",
        3: "soda_can",
    }

    image_folder = "/home/ubuntu/hackathon/src/data/"
    device = torch.device("cuda")
    layer_output_size = 512

    # prepare model and layers
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    layer = model._modules.get("avgpool")
    model = model.to(device)
    model.eval()

    # set transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
        ),
    ])

    # get list of all images in dataset
    image_paths = get_file_paths(image_folder)
    results_dict = {}
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).to(device)
        label = get_key_by_value(label_map, image_path.split("/")[-2])
        
        embedding_dim = torch.zeros(len(image), layer_output_size, 1, 1)
        
        with torch.no_grad():
            embedding_dim = model(image.unsqueeze(0))

        results_dict[image_path] = {
            'label': label,
            'embedding': embedding_dim.tolist()
        }
    # save to json
    with open('data.json', 'w') as json_file:
        json.dump(results_dict, json_file)

        



    
    
    
    


    




if __name__ == "__main__":
    main()