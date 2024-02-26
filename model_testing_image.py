import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

IMAGE_FILE = "/Users/felixpham/Documents/Seattle University/Capstone Project Info/Capstone_24.15_Soccer_Tracking/coco/test/images/scene01621_png.rf.524eaa57b77060b534865762b58d3909.jpg"

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

# Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='retrained_model2/model.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# Detect objects in the input image.
detection_result = detector.detect(image)

# Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Show the image in a window
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Object Detection", rgb_annotated_image)

# Wait for a key press or 30 milliseconds, whichever comes first
while True:
    key = cv2.waitKey(30)

    # If the Esc key (ASCII code 27) is pressed, break out of the loop
    if key == 27:
        cv2.destroyAllWindows()
        break
