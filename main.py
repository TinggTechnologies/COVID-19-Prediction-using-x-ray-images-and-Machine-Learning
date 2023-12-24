import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

model = load_model('model/model.h5')

# Create the Tkinter application window
window = tk.Tk()
window.title("COVID-19 X-ray Image Detection")

# Variable to store the file path
file_path = ""

# Function to open the file dialog and select an image
def open_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).resize((224, 224))  # Resize the image if needed
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        predict_button.configure(state=tk.NORMAL)

# Function to predict COVID-19 based on the uploaded image
def predict_covid():
    global file_path
    image = Image.open(file_path).resize((224, 224))  # Resize the image if needed
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        result = "COVID-19 Negative"
    else:
        result = "COVID-19 Positive"
    result_label.configure(text=result)

# Create the Open Image button
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack()

# Create a label to display the uploaded image
image_label = tk.Label(window)
image_label.pack()

# Create the Predict button
predict_button = tk.Button(window, text="Predict COVID-19", command=predict_covid, state=tk.DISABLED)
predict_button.pack()

# Create a label to display the prediction result
result_label = tk.Label(window, text="")
result_label.pack()

# Run the Tkinter event loop
window.mainloop()
