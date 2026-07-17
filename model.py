# Import PyTorch Library
import torch

# Import Neural Network Module
import torch.nn as nn

# Import Activation Functions
import torch.nn.functional as F


# Create CNN Model
class AnimalCNN(nn.Module):

    # Constructor
    def __init__(self, num_classes):

        # Call Parent Class Constructor
        super(AnimalCNN, self).__init__()

        # -------------------------------
        # First Convolution Layer
        # Input  : 3 Channels (RGB Image)
        # Output : 16 Feature Maps
        # Kernel : 3 × 3
        # -------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )

        # -------------------------------
        # Second Convolution Layer
        # Input  : 16 Feature Maps
        # Output : 32 Feature Maps
        # -------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        # -------------------------------
        # Max Pooling Layer
        # Kernel Size = 2 × 2
        # It reduces image size by half
        # -------------------------------
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -------------------------------
        # Fully Connected Layer
        # After two Conv + Pool layers
        # Output size becomes:
        # 32 × 54 × 54
        # Total Features = 93312
        # -------------------------------
        self.fc1 = nn.Linear(
            32 * 54 * 54,
            128
        )

        # -------------------------------
        # Output Layer
        # 128 Neurons
        # ↓
        # Number of Classes
        # -------------------------------
        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    # Forward Propagation
    def forward(self, x):

        # First Convolution
        x = self.conv1(x)

        # ReLU Activation
        x = F.relu(x)

        # Max Pooling
        x = self.pool(x)

        # Second Convolution
        x = self.conv2(x)

        # ReLU Activation
        x = F.relu(x)

        # Max Pooling
        x = self.pool(x)

        # Convert 3D Feature Maps into 1D Vector
        x = x.view(x.size(0), -1)

        # First Fully Connected Layer
        x = self.fc1(x)

        # ReLU Activation
        x = F.relu(x)

        # Final Output Layer
        x = self.fc2(x)

        return x


# -------------------------------
# Check Model
# -------------------------------
if __name__ == "__main__":

    # Suppose Dataset has 5 Classes
    model = AnimalCNN(num_classes=5)

    # Print Model Architecture
    print(model)

    # Create Dummy Input Image
    x = torch.randn(1, 3, 224, 224)

    # Forward Pass
    output = model(x)

    # Output Shape
    print("\nOutput Shape :", output.shape)