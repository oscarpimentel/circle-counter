import cv2
import numpy as np


BORDER_MODE = cv2.BORDER_CONSTANT  # BORDER_WRAP BORDER_CONSTANT BORDER_REFLECT
INTERPOLATION_METHOD = cv2.INTER_NEAREST


def resize_image(image: np.ndarray, p: float,
                 interpolation_method: int = INTERPOLATION_METHOD,
                 ) -> np.ndarray:
    if p == 1:
        return image
    image_is_single_channel = len(image.shape) == 2
    if image_is_single_channel:
        image = image[..., None]
    height, width, _ = image.shape
    new_width = int(width * p)
    new_height = int(height * p)
    new_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)
    return new_image


def get_image_filtered_by_color(original, bgr_color,
                                eps=1,
                                kernel=(5, 5),
                                iterations=2,
                                ):
    min_bgr_color = tuple([c - eps for c in bgr_color])
    max_bgr_color = tuple([c + eps for c in bgr_color])
    image = cv2.inRange(original, min_bgr_color, max_bgr_color)
    # image = (image > 10).astype(np.uint8) * 255
    se_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, se_kernel, iterations=iterations)
    return image


def get_circle_centers(image):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        m = cv2.moments(contour)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
    return centers
