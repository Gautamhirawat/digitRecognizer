import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
tf.random.set_seed(42)


testno =  2
# Load and preprocess the data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Build the model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

# Visualize predictions
def visualize_predictions(model, x_test, y_test, num_images=50):
    predictions = model.predict(x_test)
    images_per_figure = 50  # Number of images to display per figure
    num_figures = (num_images + images_per_figure - 1) // images_per_figure  # Calculate the number of figures needed

    for fig_num in range(num_figures):
        plt.figure(figsize=(10, 10))
        start_idx = fig_num * images_per_figure
        end_idx = min(start_idx + images_per_figure, num_images)
        
        for i in range(start_idx, end_idx):
            plt.subplot(5, 10, i % images_per_figure + 1)
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            plt.title(f'Pred: {np.argmax(predictions[i])}\nActual: {np.argmax(y_test[i])}')
            plt.axis('off')
            print("Done testing here" , i)
        
        plt.tight_layout()
        plt.show()

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # If the model is already trained and saved, load it
    try:
        model = load_model("mnist_cnn_model.h5")
        print("Trained model loaded successfully.")
    except OSError as e:
        print(f"Error loading model: {e}")
        # If the model is not saved yet, build and train it
        model = build_model()
        train_model(model, x_train, y_train, x_test, y_test)
        print("Model trained and saved.")
        model.save("mnist_cnn_model.h5")
        print("Model trained and saved.")

    evaluate_model(model, x_test, y_test)
    visualize_predictions(model, x_test, y_test)

    print("Model trained and saving.")
    model.save("mnist_cnn_model.h5")
    print("Model trained and saved.")


if __name__ == "__main__":
    main()