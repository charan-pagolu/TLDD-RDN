# Tomato Leaf Disease Detection Using RRDN

## Introduction
This project focuses on detecting diseases in tomato leaves using a hybrid deep learning approach. By utilizing a **Restructured Deep Residual Dense Network (RRDN)** and a pre-trained **VGG16 model**, the system efficiently classifies leaf images into six disease categories. The application includes a **Flask-based web interface** where users can upload images to receive predictions and visualize results.

---

## Project Description
The goal of this project is to identify tomato leaf diseases with high accuracy using advanced deep learning techniques. The project consists of the following components:

- **Model Development:** Trained a hybrid RRDN model with a pre-trained VGG16 backbone.
- **Web Application:** Built a Flask-based app to display results interactively.
- **Dataset:** Processed images into training and testing datasets for model evaluation.
- **Visualization:** Generated confusion matrices and performance metrics for evaluation.

---

## Summary of Tasks

### Dataset Preparation
- Resized and normalized images into 224x224 dimensions.
- Split the dataset into:
  - **Training Set:** 11,136 images
  - **Testing Set:** 2,782 images
- Six disease categories:
  1. Bacterial Spot
  2. Early Blight
  3. Healthy
  4. Leaf Mold
  5. Septoria Leaf Spot
  6. Yellow Leaf Curl Virus

### Model Implementation
- Fine-tuned the **VGG16** model for feature extraction.
- Added custom layers for classification:
  - Flatten layer
  - Dense layer (Softmax activation with 6 outputs)
- Used the **Adam optimizer** and categorical cross-entropy loss for training.

### Web Application
- Integrated Flask for building a web interface.
- Uploaded tomato leaf images are processed and classified in real-time.
- Results include:
  - Disease name
  - Prediction confidence

### Visualization
- Confusion matrix displays classification performance across categories.
- Metrics:
  - **Accuracy:** 84.8%
  - **Precision:** 95.4%
  - **Recall:** 84.8%
  - **F1-Score:** 84.8%

---

## Dataset Information
The dataset used for this project contains 13,918 images across six classes. Images are labeled and categorized into the following:

- **Training Set:** Used for model training (80% of the data).
- **Testing Set:** Used for evaluating the model's performance (20% of the data).

---

## Implementation Details

### Step 1: Data Preparation
- Resized all images to 224x224.
- Normalized pixel values using `ImageDataGenerator` for augmentation.

```python
train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=30,
    directory=train_dir,
    shuffle=True,
    target_size=(224, 224),
    class_mode='categorical'
)

### Step 2: Model Development
= Used a pre-trained VGG16 model as the base.
= Added a custom classifier for disease detection.
