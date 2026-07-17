# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

# Import our CNN Model
from model import AnimalCNN


# ==========================================================
# Check GPU or CPU
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)


# ==========================================================
# Dataset Paths
# ==========================================================

train_path = "Animal_Data/train"

validation_path = "Animal_Data/val"

test_path = "Animal_Data/test"


# ==========================================================
# Image Transform
# ==========================================================

# Resize image to 224x224
# Convert image into Tensor
# Normalize pixel values

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )

])

validation_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )

])


# ==========================================================
# Load Dataset
# ==========================================================

train_dataset = datasets.ImageFolder(

    root=train_path,

    transform=train_transform

)

validation_dataset = datasets.ImageFolder(

    root=validation_path,

    transform=validation_transform

)

test_dataset = datasets.ImageFolder(

    root=test_path,

    transform=validation_transform

)


# ==========================================================
# Class Names
# ==========================================================

class_names = train_dataset.classes

num_classes = len(class_names)

print("\nClasses")

for i, name in enumerate(class_names):

    print(i, "->", name)


# ==========================================================
# Number of Images
# ==========================================================

print("\nTraining Images   :", len(train_dataset))

print("Validation Images :", len(validation_dataset))

print("Testing Images    :", len(test_dataset))


# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=32,

    shuffle=False

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)


# ==========================================================
# Create CNN Model
# ==========================================================

model = AnimalCNN(num_classes)

# Move Model to GPU/CPU

model = model.to(device)

print("\nCNN Model Created Successfully")


# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.CrossEntropyLoss()


# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)


# ==========================================================
# Number of Epochs
# ==========================================================

epochs = 10


# ==========================================================
# Best Validation Accuracy
# ==========================================================

best_accuracy = 0

print("\nEverything is Ready...")

print("Training Started...\n")



# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(epochs):

    # Set model to training mode
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    # Loop through all training batches
    for images, labels in train_loader:

        # Move images and labels to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # -------------------------------
        # Forward Propagation
        # -------------------------------

        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)

        # -------------------------------
        # Backward Propagation
        # -------------------------------

        # Clear old gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # -------------------------------
        # Training Statistics
        # -------------------------------

        train_loss += loss.item()

        # Get predicted class
        _, predicted = torch.max(outputs, 1)

        # Total Images
        total += labels.size(0)

        # Correct Predictions
        correct += (predicted == labels).sum().item()

    # Average Training Loss
    train_loss = train_loss / len(train_loader)

    # Training Accuracy
    train_accuracy = (correct / total) * 100

    # =====================================================
    # Validation
    # =====================================================

    # Set model to evaluation mode
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    # No gradient calculation during validation
    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward Propagation
            outputs = model(images)

            # Validation Loss
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Predicted Class
            _, predicted = torch.max(outputs, 1)

            # Count Correct Predictions
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Average Validation Loss
    val_loss = val_loss / len(validation_loader)

    # Validation Accuracy
    val_accuracy = (val_correct / val_total) * 100

    # =====================================================
    # Print Result
    # =====================================================

    print("-" * 50)

    print("Epoch :", epoch + 1)

    print("Training Loss       :", round(train_loss, 4))

    print("Training Accuracy   :", round(train_accuracy, 2), "%")

    print("Validation Loss     :", round(val_loss, 4))

    print("Validation Accuracy :", round(val_accuracy, 2), "%")

    # =====================================================
    # Save Best Model
    # =====================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best Model Saved Successfully")

# ==========================================================
# Training Completed
# ==========================================================

print("\n" + "=" * 50)

print("Training Completed Successfully")

print("Best Validation Accuracy :", round(best_accuracy, 2), "%")

print("Model Saved : best_model.pth")

print("=" * 50)