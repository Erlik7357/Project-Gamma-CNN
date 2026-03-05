# Project-Gamma-CNN
CNN model for Face Mask Detection
<div align="center">

# Face Mask Detection using CNN

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

<br>

**A deep learning model to detect whether a person is wearing a face mask or not using Convolutional Neural Networks (CNN).**

![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange?style=flat-square)

---

</div>

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images into two categories:

| Class | Description |
|-------|-------------|
| **With Mask** | Person wearing a face mask |
| **Without Mask** | Person not wearing a face mask |

The model is trained on a labeled dataset of face images and can be used for real-time face mask detection.

---

## Dataset

- **Source:** [Kaggle - Face Mask Dataset](https://www.kaggle.com/datasets)
- **Classes:** 2 (With Mask, Without Mask)
- **Format:** Image files (`.jpg` / `.png`)

> The dataset was downloaded using the **Kaggle API** directly into Google Colab.

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Deep learning framework |
| **Neural Networks** | ![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat-square&logo=keras&logoColor=white) | High-level neural network API |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | Image processing & detection |
| **Numerical Computing** | ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Array operations & math |
| **Data Handling** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data manipulation |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white) | Plotting graphs & charts |
| **Dataset** | ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white) | Dataset source |
| **Environment** | ![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white) | Cloud-based notebook |

---

## Model Architecture

```
Input Image (150 x 150 x 3)
        │
        ▼
┌─────────────────────┐
│  Conv2D (32 filters)│
│   + ReLU + MaxPool  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Conv2D (64 filters)│
│   + ReLU + MaxPool  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Conv2D (128 filters)│
│   + ReLU + MaxPool  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│      Flatten        │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Dense (128) + ReLU  │
│     + Dropout (0.5)  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Dense (1) + Sigmoid │
│  (Binary Output)    │
└─────────────────────┘
```

---

## Installation

### Prerequisites

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/<your-username>/face-mask-detection-cnn.git
cd face-mask-detection-cnn
```

### Setup Kaggle API (for dataset download)

```bash
pip install kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage

### Training the Model

```python
# Run in Google Colab or Jupyter Notebook
python train_model.py
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('face_mask_model.h5')

img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (150, 150))
img = np.expand_dims(img, axis=0) / 255.0

prediction = model.predict(img)
print("With Mask" if prediction[0][0] < 0.5 else "Without Mask")
```

---

## Results

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~95%+ |
| **Validation Accuracy** | ~93%+ |
| **Loss** | Low convergence |

> *Update this section with your actual training results and include accuracy/loss curve screenshots.*

---

## Project Structure

```
face-mask-detection-cnn/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── train_model.py
├── predict.py
├── face_mask_model.h5
├── kaggle.json
├── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### If you found this project helpful, give it a star!

Made with Python & TensorFlow

</div>
