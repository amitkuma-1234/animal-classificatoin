# 🐾 Animal Image Classification using CNN (PyTorch)

## 📌 Project Description

This project implements a complete Convolutional Neural Network (CNN) for multi-class animal image classification using PyTorch.

The model is trained from scratch and supports:

- Training
- Validation
- Testing
- Model Saving
- Model Loading
- Single Image Prediction

---

# 📂 Project Structure

```
Animal_CNN_Project/

│
├── Animal_Data/
│
│   ├── Cat/
│   ├── Dog/
│   ├── Horse/
│   ├── Cow/
│   └── ...
│
├── model.py
├── train.py
├── test.py
├── predict.py
│
├── best_model.pth
├── final_model.pth
├── class_names.npy
│
├── requirements.txt
├── README.md
│
└── test.jpg
```

---
## 📦 Dataset

This repository does **not** include the dataset because it is too large for GitHub.

Please download the dataset from Kaggle:

**Animal Image Dataset (90 Different Animals):**
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

After downloading, extract the dataset and place it inside the project directory as shown below:

```text
Animal_CNN_Project/

│
├── Animal_Data/
│   ├── train/
│   │   ├── Cat/
│   │   ├── Dog/
│   │   ├── Horse/
│   │   └── ...
│   │
│   ├── test/
│   │   ├── Cat/
│   │   ├── Dog/
│   │   ├── Horse/
│   │   └── ...
│   │
│   └── val/
│       ├── Cat/
│       ├── Dog/
│       ├── Horse/
│       └── ...
│
├── model.py
├── train.py
├── test.py
├── predict.py
└── ...
```

> **Note:** The `Animal_Data` folder is excluded from this repository using `.gitignore`. Please download the dataset from Kaggle before running the project.

```

> **Note:** The `Animal_Data` folder is excluded from this repository using `.gitignore`.

# 🧠 CNN Architecture

```
Input Image (3×224×224)

↓

Conv2D (32)

↓

ReLU

↓

MaxPool

↓

Conv2D (64)

↓

ReLU

↓

MaxPool

↓

Conv2D (128)

↓

ReLU

↓

MaxPool

↓

Conv2D (256)

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Linear (512)

↓

Dropout

↓

Linear (256)

↓

Dropout

↓

Output Layer

↓

Prediction
```

---

# 📦 Dataset Structure

```
Animal_Data/

│
├── Cat/
├── Dog/
├── Horse/
├── Cow/
├── Lion/
└── ...
```

Each folder contains images belonging to a single class.

---

# 🚀 Features

- Multi-Class Image Classification
- Custom CNN Model
- Data Augmentation
- Automatic Class Detection
- GPU Support
- Validation During Training
- Best Model Saving
- Final Model Saving
- Testing Metrics
- Confusion Matrix
- Classification Report
- Single Image Prediction

---

# 🛠 Installation

Clone the repository

```
git clone https://github.com/yourusername/Animal_CNN_Project.git
```

Move into the project

```
cd Animal_CNN_Project
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶ Training

Run

```
python train.py
```

Output

```
best_model.pth

final_model.pth

class_names.npy
```

---

# 🧪 Testing

Run

```
python test.py
```

Output

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 🔍 Prediction

Place an image inside the project folder.

Example

```
test.jpg
```

Run

```
python predict.py
```

Output

```
Predicted Class

Confidence Score
```

---

# 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 💻 Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

# 📁 Output Files

```
best_model.pth

final_model.pth

class_names.npy
```

---

# 📚 Learning Objectives

- CNN Fundamentals
- Image Classification
- Data Augmentation
- Transfer Learning Ready
- Training Pipeline
- Validation Pipeline
- Testing Pipeline
- Model Saving
- Model Loading
- Image Prediction

---

# 👨‍💻 Author

Amit Kumawat

Computer Science Engineering

IIIT Vadodara

AI | Deep Learning | Computer Vision


