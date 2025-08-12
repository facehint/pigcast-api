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


# def find_largest_contour(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray', gray)
#
#     # Apply thresholding
#     _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
#
#     mask = np.ones(image.shape, dtype = np.uint8)*255
#     # Find contours
#     contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # cv2.drawContours(mask, largest_contour, -1, (0, 0, 255), thickness=cv2.FILLED)\
#     cv2.fillPoly(mask, pts=[largest_contour], color=(0 ,0, 0))
#     #
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
#     # mask = cv2.erode(mask, kernel)
#     # mask = cv2.dilate(mask, kernel)
#     # cv2.imshow('mask', mask)
#
#     # cv2.drawContours(mask, largest_contour, -1, (255, 255, 255), 40)
#     cv2.imshow('mask', mask)
#
#     condition = (mask == 255)
#     bg_image = np.ones(image.shape, dtype=np.uint8) *255
#     mark_image = np.where(condition, image, bg_image)
#     cv2.imshow('mark_image', mark_image)
#
#     alpha_channel_bg = np.ones((image.shape[0],image.shape[1],1), dtype=np.uint8)*255
#     alpha_channel_fg = np.zeros((image.shape[0],image.shape[1],1), dtype=np.uint8)
#     alpha_condition = np.all(mask==255,axis=2,keepdims=True)
#     print(alpha_condition.shape)
#
#     # print(alpha_condition)
#
#     alpha_channel = np.where(alpha_condition, alpha_channel_bg,alpha_channel_fg)
#     # print(alpha_channel_fg.shape)
#     # print(alpha_channel_bg.shape)
#     # print(alpha_channel.shape)
#
#     b, g, r = cv2.split(mark_image)
#     dst = cv2.merge([r, g, b, alpha_channel])
#     # cv2.imshow('mark_image', mark_image)
#     # cv2.imshow('alpha_channel', alpha_channel)
#     #
#     # cv2.imshow('dst', dst)
#     # Crop image by the first contour
#    #  x, y, w, h = cv2.boundingRect(largest_contour)
#    #  cropped_image = mark_image[y:y + h, x:x + w]
#    #  # cv2.imshow('cropped_image', cropped_image)
#    #
#    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#    #  # cropped_image = cv2.erode(cropped_image, kernel)
#    #  # cropped_image = cv2.dilate(cropped_image, kernel)
#    #
#    #
#    #  tmp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#    #  _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
#    #  alpha_channel = cv2.bitwise_not(alpha)
#    #  b,g,r= cv2.split(cropped_image)
#    #  # alpha_channel[:] = 0
#    #  dst = cv2.merge([r,g,b,alpha])
#     # cv2.imshow('alpha', alpha)
#     # cv2.imshow('alpha_channel', alpha_channel)
#     return dst


# Performs resizing and showing the image
# def resize_and_show(image):
#     h, w = image.shape[:2]
#     if h < w:
#         img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
#     else:
#         img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
#     # Define the specific color (in this case, blue)
#
#     # Find the largest contour of the specific color
#     largest_contour_img = find_largest_contour(img)
#
#     # Display the image
#     # cv2.imshow('123', largest_contour_img)
#     cv2.imwrite('output.png', largest_contour_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Preview the image(s)
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
    print(name)
    # resize_and_show(image)

BG_COLOR = (0, 0, 0)  # gray是
MASK_COLOR = (192, 192, 192)  # w white
# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='bighead/selfie_multiclass_256x256.tflite')
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

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    # Loop through demo image(s)
    for image_file_name in IMAGE_FILENAMES:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image.create_from_file(image_file_name)

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

        b, g, r = cv2.split(mark_image)
        dst = cv2.merge([r, g, b, alpha_channel])
        # cv2.imshow('dst', dst)
        #Crop image by the first contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = dst[y:y + h, x:x + w]
        # cv2.imshow('cropped_image', cropped_image)
        # cv2.imwrite('cropped_image.png', cropped_image)

        resize_cropped_image = resize_with_pad(cropped_image, (400, 400))
        cv2.imshow('resize_cropped_image', resize_cropped_image)
        cv2.imwrite('bighead/resize_cropped_image.png', resize_cropped_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f'Segmentation mask of {name}:')
        # resize_and_show(output_image)
