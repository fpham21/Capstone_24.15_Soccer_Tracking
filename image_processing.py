import cv2
import numpy as np

# Load the image
image = cv2.imread('images/test3.png')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set the brightness range (adjust as needed)
lower_brightness = 100
upper_brightness = 200

# Create a binary mask for pixels within the brightness range
binary_mask = cv2.inRange(gray_image, lower_brightness, upper_brightness)

# Create a black image
result_image = np.zeros_like(image)

# Set white color to the white areas in the result image
result_image[binary_mask == 255] = [255, 255, 255]  # White color in RGB format

# Save the result
cv2.imwrite('black_white.jpg', result_image)

print(result_image)
print(np.size(image))
print(np.shape(image))

cv2.imshow('field',cv2.imread('images/field.jpg'))
cv2.imshow('red_white_areas.jpg', result_image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#mediapipe for ML

#mediapipe, tensorflow 