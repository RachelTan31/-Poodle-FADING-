import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class PoodleAgeDataset(Dataset):
    """
    Loads poodle images and corresponding ages from combined.txt
    Format expected for each line:  <image_filename> <age>
    Example:  poodle_001.jpg  6.5
    """

    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load and parse combined.txt
        self.samples = []
        with open(labels_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, age_str = line.split()
                self.samples.append((img_name, float(age_str)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, age = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([age], dtype=torch.float32)


def create_model():
    """
    Loads a pretrained ResNet18 and replaces the final layer
    for single-value regression output.
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # regression output
    return model


def train(model, train_loader, device, epochs=10, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")


def main():
    images_dir = r"C:\Users\maryl\Downloads\-Poodle-FADING-\dataset\poodle_images"
    labels_file = r"C:\Users\maryl\Downloads\-Poodle-FADING-\dataset\combined.txt"
    batch_size = 16
    epochs = 20
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset + Loader
    dataset = PoodleAgeDataset(images_dir, labels_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = create_model().to(device)

    # Train
    train(model, dataloader, device, epochs=epochs, lr=lr)

    # Save model
    torch.save(model.state_dict(), "poodle_age_predictor.pt")
    print("Model saved as poodle_age_predictor.pt")


if __name__ == "__main__":
    main()
