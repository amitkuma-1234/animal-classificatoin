# ==========================================================
# Import Libraries
# ==========================================================

import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import CNN Model
from model import AnimalCNN


# ==========================================================
# Check GPU or CPU
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)


# ==========================================================
# Test Dataset Path
# ==========================================================

test_path = "Animal_Data/test"


# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.5,0.5,0.5],

        std=[0.5,0.5,0.5]

    )

])


# ==========================================================
# Load Test Dataset
# ==========================================================

test_dataset = datasets.ImageFolder(

    root=test_path,

    transform=transform

)

# Class Names

class_names = test_dataset.classes

num_classes = len(class_names)

print("\nClasses")

for i,name in enumerate(class_names):

    print(i,"->",name)


# ==========================================================
# DataLoader
# ==========================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)


# ==========================================================
# Create Model
# ==========================================================

model = AnimalCNN(num_classes)

model = model.to(device)


# ==========================================================
# Load Saved Model
# ==========================================================

model.load_state_dict(

    torch.load(

        "best_model.pth",

        map_location=device

    )

)

print("\nBest Model Loaded Successfully")


# ==========================================================
# Evaluation Mode
# ==========================================================

model.eval()


# ==========================================================
# Lists for Prediction
# ==========================================================

true_labels = []

predicted_labels = []


# ==========================================================
# Testing
# ==========================================================

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        # Forward Pass

        outputs = model(images)

        # Predicted Class

        _,predicted = torch.max(outputs,1)

        # Save Labels

        true_labels.extend(labels.cpu().numpy())

        predicted_labels.extend(predicted.cpu().numpy())


# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(

    true_labels,

    predicted_labels

)

print("\nTest Accuracy :", round(accuracy*100,2),"%")



# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(

    classification_report(

        true_labels,

        predicted_labels,

        target_names=class_names

    )

)


# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nConfusion Matrix\n")

cm = confusion_matrix(

    true_labels,

    predicted_labels

)

print(cm)


print("\nTesting Completed Successfully")