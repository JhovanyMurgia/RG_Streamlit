# ocr/extract_rg_type3.py

import re
import os
import cv2
from rapidfuzz.distance import Levenshtein

from utils.image_utils import calculate_base_angle, average_angles_boxes, rotate_image, group_words_by_lines
from config.ocr_model_config import run_ocr

def pipeline_ocr(model, image_path, confidence_threshold=0.5, show_image=False, debug=False):
    result = run_ocr(model, image_path, show_image)
    meta_data = result

    angle_list = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                angle = calculate_base_angle(word.geometry)
                if -4 < angle < 4:
                    angle_list.append(angle)

    mean_angle = average_angles_boxes(angle_list) if angle_list else 0

    if abs(mean_angle) > 1:
        rotated_image = rotate_image(image_path, mean_angle)
        cv2.imwrite("rotated_image.png", rotated_image)
        image_path = "rotated_image.png"
        result = run_ocr(model, image_path, show_image)
        os.remove("rotated_image.png")

    words_data = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > confidence_threshold:
                    words_data.append({"text": word.value, "geometry": word.geometry})

    lines = group_words_by_lines(words_data)

    if debug:
        for i, line in enumerate(lines):
            print(f"Line {i + 1}: {line}")

    return lines, meta_data

def find_most_similar_word(target, lines):
    best_word = None
    best_distance = float("inf")
    best_index = -1

    for i, text in enumerate(lines):
        for word in text.split():
            distance = Levenshtein.distance(target.upper(), word.upper())
            if distance < best_distance:
                best_distance = distance
                best_word = word
                best_index = i

    return best_distance, best_word, best_index

def extract_name(lines):
    lines_cleaned = [line.replace("/", " ") for line in lines]

    dist_pt, _, line_pt = find_most_similar_word("NOME", lines_cleaned)
    dist_en, _, line_en = find_most_similar_word("NAME", lines_cleaned)

    line_index = line_en if dist_en < dist_pt else line_pt
    name = lines[line_index + 1] if line_index + 1 < len(lines) else ""
    name = re.sub(r'[.,-/]', '', name).strip()

    return name

def extract_parents(lines):
    lines_cleaned = [line.replace("/", " ") for line in lines]

    dist_pt, _, line_pt = find_most_similar_word("FILIAÇAO", lines_cleaned)
    dist_en, _, line_en = find_most_similar_word("FILIATION", lines_cleaned)
    start_line = line_en if dist_en < dist_pt else line_pt

    dist1, _, end1 = find_most_similar_word("EXPEDIDOR", lines_cleaned)
    dist2, _, end2 = find_most_similar_word("CARD", lines_cleaned)
    end_line = end2 if dist2 < dist1 else end1

    if start_line + 1 < len(lines) and end_line < len(lines):
        parents_text = "  ".join(lines[start_line + 1:end_line])
        parents_text = re.sub(r'\b[a-zA-Z]*\d{2,12}[a-zA-Z]*\b|\b\d{1,12}\b|\b[a-zA-Z]\b|[.\-/]', '', parents_text)
        parents_text = parents_text.strip().replace("    ", "  ").replace("  ", " E ").replace(" E  E ", " E ")
        return parents_text.strip()
    return None

def extract_cpf(lines):
    patterns = [
        r'CPF[\s:]*([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})',
        r'\b([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})\b'
    ]
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return None

def extract_date(lines):
    patterns = [
        r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b',
        r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b',
        r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b'
    ]
    for pattern in patterns:
        for line in lines:
            match = re.search(pattern, line.strip())
            if match:
                return match.group(1)
    return None

def extract_rg_t3(model, front_path, back_path, confidence_threshold=0.5, show_image=False, debug=False):
    try:
        lines_front, meta_front = pipeline_ocr(model, front_path, confidence_threshold, show_image, debug)
        name = extract_name(lines_front)
        cpf = extract_cpf(lines_front)
        birth_date = extract_date(lines_front)
    except FileNotFoundError:
        print(f"[ERROR] Front image not found: {front_path}")
        name = cpf = birth_date = None

    try:
        lines_back, meta_back = pipeline_ocr(model, back_path, confidence_threshold, show_image, debug)
        parents = extract_parents(lines_back)
        issue_date = extract_date(lines_back)
    except FileNotFoundError:
        print(f"[ERROR] Back image not found: {back_path}")
        parents = issue_date = None

    data = {
        "Número do RG": cpf,
        "CPF": cpf,
        "Nome": name,
        "Filiação": parents,
        "Data de expedição": issue_date,        
        "Data de nascimento": birth_date
    }

    return data, meta_front, meta_back