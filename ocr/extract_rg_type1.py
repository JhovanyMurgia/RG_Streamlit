# ocr/extract_rg_type1.py

import re
import os
import cv2
from rapidfuzz.distance import Levenshtein

from utils.image_utils import calculate_base_angle, average_angles_boxes, rotate_image, group_words_by_lines
from config.ocr_model_config import run_ocr

# OCR pipeline

def pipeline_ocr(model, image_path, confidence_threshold=0.5, show_image=False, debug=False):
    result = run_ocr(model, image_path, show_image)
    meta_data = result

    angle_list = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                angle = calculate_base_angle(word.geometry)
                if -10 < angle < 10:
                    angle_list.append(angle)

    if len(angle_list) > 0:
        mean_angle = average_angles_boxes(angle_list)
    else:
        mean_angle = 0

    if mean_angle > 1 or mean_angle < -1:
        rotated_image = rotate_image(image_path, mean_angle)
        cv2.imwrite("rotated_image.jpg", rotated_image)
        image_path = "rotated_image.jpg"
        result = run_ocr(model, image_path, show_image)
        os.remove("rotated_image.jpg")

    words_data = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > confidence_threshold:
                    words_data.append({
                        "text": word.value,
                        "geometry": word.geometry
                    })

    lines = group_words_by_lines(words_data)

    if debug:
        for i, line_text in enumerate(lines):
            print(f"Line {i + 1}: {line_text}")

    return lines, meta_data

# Text extraction functions for the new RG

def find_most_similar_word(target, lines):
    best_word = None
    lowest_distance = float("inf")
    best_index = -1

    for i, text in enumerate(lines):
        for word in text.split():
            distance = Levenshtein.distance(target.upper(), word.upper())
            if distance < lowest_distance:
                lowest_distance = distance
                best_word = word
                best_index = i

    return best_word, best_index

def extract_name_and_parents(lines):
    name_word, name_index = find_most_similar_word("NOME", lines)
    parent_word, parent_index = find_most_similar_word("FILIAÇAO", lines)
    date_word, date_index = find_most_similar_word("DATA", lines)

    name = ' '.join(lines[name_index:parent_index]).replace(name_word, '').strip()
    parents = ' '.join(lines[parent_index:date_index]).replace(parent_word, '').strip()

    return name, parents

def extract_new_rg_number(lines):
    rg_patterns = [
        r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})',
        r'\b([0-9]{2,3}\.[0-9]{3}\.[0-9]{3}-[0-9]{1,2})\b',
        r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b',
        r'\b([0-9]{1,2}\s[0-9]{3}\s[0-9]{3})\b',
        r'\b([0-9]{3}\.[0-9]{3})\b'
    ]

    for pattern in rg_patterns:
        regex = re.compile(pattern)
        for line in lines:
            match = regex.search(line.strip())
            if match:
                return match.group(1)
    return None

def extract_cpf_number(lines):
    cpf_patterns = [
        r'CPF[\s:]*([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})',
        r'\b([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})\b'
    ]

    for i, line in enumerate(lines):
        for pattern in cpf_patterns:
            match = re.search(pattern, line)
            if match:
                cpf = match.group(1)
                rg = extract_new_rg_number(lines[i + 1:])
                return cpf, rg
    return None, None

def extract_date(lines):
    date_patterns = [
        r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b',
        r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b',
        r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b'
    ]

    for pattern in date_patterns:
        for line in lines:
            match = re.search(pattern, line.strip())
            if match:
                return match.group(1)
    return None

def extract_rg_t1(model, front_path, back_path, confidence_threshold=0.5, show_image=False, debug=False):
    try:
        lines_front, metadata_front = pipeline_ocr(model, front_path, confidence_threshold, show_image, debug)
        name, parents = extract_name_and_parents(lines_front)
        birth_date = extract_date(lines_front)
    except FileNotFoundError:
        print(f"Image not found: {front_path}")
        name = parents = birth_date = None

    try:
        lines_back, metadata_back = pipeline_ocr(model, back_path, confidence_threshold, show_image, debug)
        cpf, rg = extract_cpf_number(lines_back)
        if not rg:
            rg = extract_new_rg_number(lines_back)
        issue_date = extract_date(lines_back)
    except FileNotFoundError:
        print(f"Image not found: {back_path}")
        cpf = rg = issue_date = None

    extracted_data = {
        "Número do RG": rg,
        "CPF": cpf,
        "Nome": name,
        "Filiação": parents,
        "Data de expedição": issue_date,        
        "Data de nascimento": birth_date
    }

    return extracted_data, metadata_front, metadata_back
