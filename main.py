import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Path to the test image
image_path = 'test\images1.jpeg'

# Parameters
IMG_SIZE = 28  # Image size used for training

def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 28x28
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    img_expanded = np.expand_dims(img_normalized, axis=(0, -1))  # Add batch and channel dimensions
    return img_expanded, img_resized

def predict_number(image_path):
    """
    Predict the digit in the input image.
    """
    img_preprocessed, img_original = preprocess_image(image_path)
    prediction = model.predict(img_preprocessed, verbose=0)  # Predict using the model
    predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability
    return predicted_digit, img_original

if __name__ == "__main__":
    # Predict the digit
    digit, processed_image = predict_number(image_path)

    # Display the result
    plt.imshow(processed_image, cmap='gray')
    plt.title(f"Recognized Digit: {digit}")
    plt.axis('off')
    plt.show()

    print(f"The recognized digit is: {digit}")
