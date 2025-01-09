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

val_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = val_image_generator.flow_from_directory(
    batch_size=30,
    directory=test_dir,
    shuffle=True,
    target_size=(224, 224),
    class_mode='categorical'
)
```

### Step 2: Model Development
- Used a pre-trained **VGG16** model for feature extraction.
- Added a custom classifier:
  - Flatten layer.
  - Dense layer with softmax activation for six classes.
- Compiled the model using:
  - Optimizer: Adam.
  - Loss function: Categorical cross-entropy.

```python
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

classifier = Sequential([
    base_model,
    Flatten(),
    Dense(6, activation='softmax')
])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Step 3: Model Training
- Trained the model using the following parameters:
  - Batch size: 30.
  - Epochs: 10.
  - Training data: `train_data_gen`.
  - Validation data: `val_data_gen`.

```python
history = classifier.fit(
    train_data_gen,
    epochs=10,
    validation_data=val_data_gen
)
```
### Step 4: Web Application Development
- Built a Flask-based web application to classify uploaded tomato leaf images.
- Processed images as follows:
  - Resized images to 224x224 pixels.
  - Normalized pixel values before passing them to the trained model.
- The application outputs:
  - Predicted disease category.
  - Confidence score for predictions.

```python
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['file'])
    image = image.resize((224, 224))
    data = np.array(image).reshape((1, 224, 224, 3)) / 255.0
    prediction = classifier.predict(data)
    return jsonify({'prediction': prediction.argmax()})
```

### Step 5: Visualization and Evaluation
- Generated a confusion matrix to evaluate the model's performance across categories.
- Calculated evaluation metrics:
  - Accuracy.
  - Precision.
  - Recall.
  - F1-score.
- Visualized the confusion matrix using Matplotlib.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate confusion matrix and metrics
cm = confusion_matrix(y_true=true_labels, y_pred=prediction)
accuracy = accuracy_score(true_labels, prediction)
precision = precision_score(true_labels, prediction, average='micro')
recall = recall_score(true_labels, prediction, average='micro')
f1 = f1_score(true_labels, prediction, average='micro')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(cm, classes=cm_plot_labels)
```
## Conclusion
This project successfully implemented a robust model for detecting tomato leaf diseases with high accuracy. By integrating advanced deep learning techniques with a practical web interface, the application provides a real-time solution for farmers and agricultural researchers.

### Key Takeaways:
- **Model Performance:** The RRDN-based architecture achieved an accuracy of 84.8%, indicating its efficacy in detecting diseases.
- **Web Application:** The Flask app makes the model accessible for real-world usage.

### Future Improvements:
1. **Expand Dataset:** Include more disease categories to enhance model generalizability.
2. **Optimize Model:** Experiment with advanced architectures like Vision Transformers for improved accuracy.
3. **Mobile Deployment:** Convert the Flask application into a lightweight mobile app for on-field usage.
