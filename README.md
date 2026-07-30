# ✏️ MNIST Digit Predictor using ANN

A handwritten digit recognition application built with TensorFlow, Keras, and Tkinter. Draw a digit on the canvas, and an Artificial Neural Network (ANN) will predict which number it is.

## 📖 About

This project demonstrates image classification using an Artificial Neural Network trained on the MNIST handwritten digit dataset. Users can draw a digit (0–9) in the application, and the trained model predicts the digit along with its confidence score.

The application also allows users to clear the canvas and save their drawings as image files.

## ✨ Features

* Artificial Neural Network (ANN) classifier
* Interactive drawing canvas
* Real-time digit prediction
* Prediction confidence score
* Save drawings as PNG images
* Clear canvas with one click
* Simple and user-friendly interface

## 🛠 Technologies

* Python
* TensorFlow / Keras
* Tkinter
* NumPy
* Pillow (PIL)

## 🧠 Model

The neural network consists of:

* Flatten input layer (28×28)
* Dense layer (256 neurons, ReLU)
* Dense layer (128 neurons, ReLU)
* Dense layer (64 neurons, ReLU)
* Output layer (10 neurons, Softmax)

The model is trained to recognize handwritten digits from **0 to 9**.

## 🚀 How to Run

1. Clone the repository.

```bash
git clone https://github.com/yourusername/mnist-digit-predictor-ann.git
```

2. Install the required libraries.

```bash
pip install tensorflow pillow numpy
```

3. Make sure `best_weights.weights.h5` is in the project folder.

4. Run the application.

```bash
python app.py
```

## 📷 How It Works

1. Draw a digit on the canvas.
2. Click **Predict Digit**.
3. The ANN predicts the digit and displays its confidence.
4. Optionally save your drawing or clear the canvas.

## 📂 Dataset

The model was trained using the **MNIST Handwritten Digits** dataset, a widely used benchmark dataset containing 70,000 grayscale images of handwritten digits.
