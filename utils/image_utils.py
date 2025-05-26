# utils/image_utils.py

import math
import cv2

def calculate_base_angle(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        return angle_deg + 0.5
    elif angle_deg > 0:
        return angle_deg - 0.5
    else:
        return angle_deg

def average_angles_boxes(angle_list):
    sorted_list = sorted(angle_list)
    n = len(sorted_list)
    middle = n // 2

    if n < 4:
        return sum(angle_list) / len(angle_list)

    if n % 2 == 1:
        central_values = sorted_list[middle - 1:middle + 3]
    else:
        central_values = sorted_list[middle - 2:middle + 2]

    return sum(central_values) / 4

def rotate_image(image_path, angle):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LANCZOS4)

    return rotated_image

def get_y_center(geometry):
    top_y = min(coord[1] for coord in geometry)
    bottom_y = max(coord[1] for coord in geometry)
    return (top_y + bottom_y) / 2

def group_words_by_lines(words_data, tolerance=0.01):
    for word in words_data:
        word["y_center"] = get_y_center(word["geometry"])

    words_data.sort(key=lambda w: w["y_center"])

    lines = []
    current_line = []
    last_y = None

    for word in words_data:
        if last_y is None or abs(word["y_center"] - last_y) <= tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
        last_y = word["y_center"]

    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda w: min(coord[0] for coord in w["geometry"]))

    return [' '.join(word['text'] for word in line) for line in lines]
