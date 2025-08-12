import numpy as np
import mediapipe as mp
from typing import Tuple

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
# from google.colab.patches import cv2_imshow
import math

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# IMAGE_FILENAMES = ["two_person.png"]
# IMAGE_FILENAMES = ["baby_with_glass_long_hair1.jpg"]
IMAGE_FILENAMES = ["child_with_glass.jpeg"]
# IMAGE_FILENAMES = ["baby_with_glass_long_hair2.jpg"]
# IMAGE_FILENAMES = ["baby_with_hat.jpg"]
# IMAGE_FILENAMES = ["baby_with_glass.jpg"]

import cv2
import numpy as np

BG_COLOR = (0, 0, 0)  # gray是
MASK_COLOR = (192, 192, 192)  # w white





def find_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # Apply thresholding
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


BG_COLOR = (0, 0, 0)  # gray是
MASK_COLOR = (192, 192, 192)  # w white
# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='./selfie_multiclass_256x256.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)
def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def crop(contents:bytes)-> bytes:
    nparr = np.frombuffer(contents, np.uint8)
    cv2_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_image)

    segmenter =  vision.ImageSegmenter.create_from_options(options)

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()

    bg_image = np.ones(image_data.shape, dtype=np.uint8) * 0

    segment_category = np.stack((category_mask.numpy_view(),) * 3, axis=-1)

    condition = (segment_category == 3) | (segment_category == 1) | (segment_category == 5)
    # condition = segment_category >0.1
    output_image = np.where(condition, image_data, bg_image)
    largest_contour = find_largest_contour(output_image)

    largest_contour_mask = np.ones(image_data.shape, dtype=np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    # mask = cv2.erode(mark_image, kernel)
    largest_contour_mask = cv2.dilate(largest_contour_mask, kernel)

    cv2.fillPoly(largest_contour_mask, pts=[largest_contour], color=(0, 0, 0))

    condition_mask = (largest_contour_mask == 0)
    bg_image = np.ones(image_data.shape, dtype=np.uint8) * 255
    mark_image = np.where(condition_mask, image_data, bg_image)
    # cv2.imshow('mark_image', mark_image)

    cv2.drawContours(mark_image, largest_contour, -1, (240, 240, 240), 3)

    alpha_channel_bg = np.ones((image_data.shape[0], image_data.shape[1], 1), dtype=np.uint8) * 255
    alpha_channel_fg = np.zeros((image_data.shape[0], image_data.shape[1], 1), dtype=np.uint8)
    alpha_condition = np.all(largest_contour_mask == 0, axis=2, keepdims=True)

    alpha_channel = np.where(alpha_condition, alpha_channel_bg, alpha_channel_fg)

    r, g, b = cv2.split(mark_image)
    dst = cv2.merge([r, g, b, alpha_channel])
    # cv2.imshow('dst', dst)
    # Crop image by the first contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = dst[y:y + h, x:x + w]
    # cv2.imshow('cropped_image', cropped_image)
    # cv2.imwrite('cropped_image.png', cropped_image)

    resize_cropped_image = resize_with_pad(cropped_image, (400, 400))
    # cv2.imshow('resize_cropped_image', resize_cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    success, encoded_image = cv2.imencode('.png', resize_cropped_image)
    return encoded_image.tobytes()

if __name__ == '__main__':
    image = mp.Image.create_from_file("child_with_glass.jpeg")
    result = crop(image)
    cv2.imshow('resize_cropped_image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Create the image segmenter
# with vision.ImageSegmenter.create_from_options(options) as segmenter:
#     # Loop through demo image(s)
#     for image_file_name in IMAGE_FILENAMES:
#         # Create the MediaPipe image file that will be segmented
#
#         cv2.imshow('resize_cropped_image', resize_cropped_image)
#         cv2.imwrite('bighead/resize_cropped_image.png', resize_cropped_image)
#
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
