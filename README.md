# 🌿 Plant Disease Detection System

A deep learning-based plant disease detection system that classifies **14 disease classes** across common crops using a CNN (ResNet9 architecture) trained on a subset of the PlantVillage dataset.

---

## 📁 Project Structure

```
plant_disease_detection/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters & paths
│
├── data_prep/
│   ├── download_dataset.py      # Download small subset from Kaggle
│   └── data_loader.py           # Dataset class, transforms, DataLoaders
│
├── models/
│   ├── resnet9.py               # ResNet9 model architecture
│   └── train.py                 # Training loop with accuracy/loss tracking
│
├── utils/
│   ├── evaluate.py              # Evaluate model, confusion matrix, metrics
│   ├── predict.py               # Single image prediction utility
│   └── visualize.py             # Plot training curves, sample images
│
├── app.py                       # Streamlit web application
├── train_model.py               # Main script to run full training pipeline
└── disease_info.json            # Disease descriptions & remedies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
python data_prep/download_dataset.py
```
> This downloads a **small subset** (~200MB) with 14 classes from the PlantVillage dataset using the Kaggle API.

### 3. Train the model
```bash
python train_model.py
```

### 4. Run the web app
```bash
streamlit run app.py
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet9 (custom lightweight) |
| Dataset | PlantVillage (14-class subset) |
| Image Size | 224×224 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| LR Scheduler | OneCycleLR |

---

## 🌱 Supported Classes (14)

| # | Class |
|---|-------|
| 1 | Apple - Apple Scab |
| 2 | Apple - Black Rot |
| 3 | Apple - Healthy |
| 4 | Corn - Common Rust |
| 5 | Corn - Healthy |
| 6 | Grape - Black Rot |
| 7 | Grape - Healthy |
| 8 | Potato - Early Blight |
| 9 | Potato - Late Blight |
| 10 | Potato - Healthy |
| 11 | Tomato - Bacterial Spot |
| 12 | Tomato - Early Blight |
| 13 | Tomato - Late Blight |
| 14 | Tomato - Healthy |

---

## 📊 Training Results (Expected)
- Validation Accuracy: ~90–95%
- Training Time: ~10–20 mins on GPU, ~30–60 mins on CPU

---

## 💡 Features
- ✅ Custom ResNet9 model (fast, lightweight)
- ✅ Data augmentation (flip, rotate, color jitter)
- ✅ Training plots (loss & accuracy curves)
- ✅ Confusion matrix visualization
- ✅ Streamlit web app for live predictions
- ✅ Disease info + remedies for each class
