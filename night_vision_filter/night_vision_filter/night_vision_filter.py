import cv2
import numpy as np
import os 
from PIL import Image 
import os 

def night_vision_filter(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization for enhancing low-light details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_image = clahe.apply(gray_image)
    
    # Apply a green tint to simulate night vision
    green_tint = np.zeros_like(image)
    green_tint[:,:,1] = equalized_image
    
    # Add noise to the image to mimic night vision effect
    noise = np.random.normal(0, 25, image.shape)
    noisy_image = cv2.add(green_tint.astype(np.float64), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# Load an example image
image = cv2.imread('me.jpg')
image = cv2.imread("test001.jpg")

# Apply the night vision filter
night_vision_image = night_vision_filter(image)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.waitKey(0)  # Wait for a key press before displaying the next image
cv2.imshow('Night Vision Filter', night_vision_image)
cv2.waitKey(0)  # Wait for a key press before closing the windows
cv2.destroyAllWindows()
