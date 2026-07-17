# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

# Import CNN Model
from model import AnimalCNN


# ==========================================================
# Check GPU or CPU
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)


# ==========================================================
# Get Class Names
# ==========================================================

# ImageFolder automatically reads class names
dataset = datasets.ImageFolder("Animal_Data/train")

class_names = dataset.classes

num_classes = len(class_names)

print("\nClasses")

for i, name in enumerate(class_names):

    print(i, "->", name)


# ==========================================================
# Create CNN Model
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
# Image Path
# ==========================================================

# Change image name here

image_path = "test.jpg"


# ==========================================================
# Read Image
# ==========================================================

image = Image.open(image_path)

image = image.convert("RGB")


# ==========================================================
# Display Image
# ==========================================================

plt.imshow(image)

plt.axis("off")


# ==========================================================
# Image Preprocessing
# ==========================================================

image = transform(image)

# Add Batch Dimension
# Shape:
# (3,224,224)
# ↓
# (1,3,224,224)

image = image.unsqueeze(0)

image = image.to(device)


# ==========================================================
# Prediction
# ==========================================================

with torch.no_grad():

    outputs = model(image)

    probabilities = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)


# ==========================================================
# Result
# ==========================================================

predicted_class = class_names[predicted.item()]

confidence = confidence.item() * 100


print("\nPrediction Result")

print("---------------------------")

print("Predicted Class :", predicted_class)

print("Confidence      : {:.2f}%".format(confidence))


# ==========================================================
# Show Prediction on Image
# ==========================================================

plt.title(

    predicted_class +

    "\nConfidence : {:.2f}%".format(confidence)

)

plt.show()