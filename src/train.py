import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.datasets as datasets

from dataset import ImageEmbeddings
"""
This does not working
"""

def count_folders_in_directory(directory):
    folder_count = 0
    for _, dirs, _ in os.walk(directory):
        folder_count += len(dirs)
    return folder_count

def main() -> None:
    device = torch.device("cuda")
    data_root = "/home/ubuntu/hackathon/src/data/"
    batch_size = 32
    num_epochs = 10
    lr = 0.001
    momentum = 0.9
    num_classes = count_folders_in_directory(data_root)

    # instantiate a model with a resnet18 backbone
    # and remove the classification layer since this
    # is not a classification task. We are only
    # interested in the embeddings
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    for epoch in range(num_epochs):
        running_loss = 0
        correct_predictions = 0

        for images, labels, in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / (len(dataloader) * batch_size)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")



if __name__ == "__main__":
    main()