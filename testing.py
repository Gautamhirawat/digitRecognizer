from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_photo(photo_path):
    # Load the photo
    img = Image.open(photo_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    return img_array

def predict_photo(model, photo_path):
    img_array = preprocess_photo(photo_path)
    prediction = model.predict(img_array)
    print(f'Prediction array: {prediction}')
    predicted_class = np.argmax(prediction)
    return predicted_class



# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Ask the user to choose an image file
photo_path = "/content/dummy.png" # if you are using .py insted of ipynb use photo_path = "image.png"

if photo_path:
    # Predict the class of the chosen photo
    predicted_class = predict_photo(model, photo_path)
    print(f'The predicted class is: {predicted_class}')
else:
    print("No file selected.")