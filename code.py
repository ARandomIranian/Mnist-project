import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os

# =========================
# MODEL
# =========================
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.build((None, 28, 28))
model.load_weights("best_weights.weights.h5")

print("Model loaded successfully!")

# =========================
# UI COLORS
# =========================
COLOR_BG = "#FDF6F0"
COLOR_CANVAS = "#FFFFFF"
COLOR_PRIMARY = "#FF9AA2"
COLOR_HOVER = "#FFB7B2"
COLOR_TEXT = "#4A4A4A"

# =========================
# WINDOW
# =========================
screen = tk.Tk()
screen.geometry("500x650")
screen.title("MNIST Digit Predictor")
screen.config(bg=COLOR_BG)

title = tk.Label(
    screen,
    text="Draw a Digit",
    font=("Helvetica", 20, "bold"),
    bg=COLOR_BG,
    fg=COLOR_PRIMARY
)
title.pack(pady=20)

# =========================
# CANVAS
# =========================
canvas_frame = tk.Frame(screen, bg="#E0E0E0", padx=2, pady=2)
canvas_frame.pack()

paint_screen = tk.Canvas(
    canvas_frame,
    width=300,
    height=300,
    bg=COLOR_CANVAS,
    highlightthickness=0
)
paint_screen.pack()

x_start = 0
y_start = 0

def start_draw(event):
    global x_start, y_start
    x_start = event.x
    y_start = event.y

def draw(event):
    global x_start, y_start

    paint_screen.create_line(
        x_start, y_start,
        event.x, event.y,
        width=10,              # thicker = better MNIST results
        fill="black",          # IMPORTANT FIX
        capstyle="round",
        smooth=True
    )

    x_start = event.x
    y_start = event.y

paint_screen.bind("<Button-1>", start_draw)
paint_screen.bind("<B1-Motion>", draw)

# =========================
# SAVE FUNCTION
# =========================
def save_canvas():
    image = Image.new("L", (300, 300), 255)
    draw_obj = ImageDraw.Draw(image)

    for item in paint_screen.find_all():
        coords = paint_screen.coords(item)
        if len(coords) >= 4:
            draw_obj.line(coords, fill=0, width=10)

    i = 1
    while os.path.exists(f"sketch_{i}.png"):
        i += 1

    filename = f"sketch_{i}.png"
    image.save(filename)

    result_label.config(text=f"Saved: {filename}")

# =========================
# PREDICTION (FIXED INVERSION)
# =========================
def predict_digit():
    image = Image.new("L", (300, 300), 255)
    draw_obj = ImageDraw.Draw(image)

    for item in paint_screen.find_all():
        coords = paint_screen.coords(item)
        if len(coords) >= 4:
            draw_obj.line(coords, fill=0, width=10)

    # resize to MNIST format
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(image)

    # INVERSION FIX (white background -> black background model expects)
    img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {digit} ({confidence:.1f}%)"
    )

# =========================
# CLEAR
# =========================
def clear_canvas():
    paint_screen.delete("all")
    result_label.config(text="Prediction: ?")

# =========================
# BUTTONS
# =========================
predict_btn = tk.Button(
    screen,
    text="Predict Digit",
    command=predict_digit,
    bg="#A8E6CF",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    padx=20,
    pady=10
)
predict_btn.pack(pady=10)

save_btn = tk.Button(
    screen,
    text="Save My Sketch",
    command=save_canvas,
    bg=COLOR_PRIMARY,
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    padx=20,
    pady=10,
    activebackground=COLOR_HOVER
)
save_btn.pack(pady=10)

clear_btn = tk.Button(
    screen,
    text="Clear",
    command=clear_canvas,
    bg="#FFD3B6",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    padx=20,
    pady=10
)
clear_btn.pack(pady=10)

# =========================
# RESULT LABEL
# =========================
result_label = tk.Label(
    screen,
    text="Prediction: ?",
    font=("Helvetica", 18, "bold"),
    bg=COLOR_BG,
    fg=COLOR_TEXT
)
result_label.pack(pady=20)

screen.mainloop()
