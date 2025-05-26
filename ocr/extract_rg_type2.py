# ocr/extract_rg_type2.py

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

    lines = group_words_by_lines(words_data, tolerance=0.02)

    if debug:
        for i, line_text in enumerate(lines):
            print(f"Line {i + 1}: {line_text}")

    return lines, meta_data

def find_best_text(target, lines):
    best_text = None
    lowest_distance = float("inf")
    best_index = -1
    similarity = 0.0

    for i, text in enumerate(lines):
        for word in text.split():
            distance = Levenshtein.distance(target.upper(), word.upper())
            if distance < lowest_distance:
                lowest_distance = distance
                best_text = word
                best_index = i
                similarity = (1 - distance / max(len(target), len(word))) * 100

    return best_text, best_index, similarity

def extract_rg_number(lines):
    rg_patterns = [
        r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})',
        r'\b([0-9]{2,3}\.[0-9]{3}\.[0-9]{3}-[0-9]{1,2})\b',
        r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b',
        r'\b([0-9]{1,2}\s[0-9]{3}\s[0-9]{3})\b',
        r'\b([0-9]{3}\.[0-9]{3})\b'
    ]

    for pattern in rg_patterns:
        regex = re.compile(pattern)
        for line in lines[:4]:
            match = regex.search(line.strip())
            if match:
                return match.group(1)
    return None

def extract_cpf(lines):
    cpf_pattern = re.compile(r'\b([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})\b')
    for line in lines[3:]:
        match = cpf_pattern.search(line.strip())
        if match:
            return match.group(1)
    return None

def extract_date(lines, start=0):
    date_patterns = [
        r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b',
        r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b',
        r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b'
    ]

    for pattern in date_patterns:
        regex = re.compile(pattern)
        for line in lines[start:]:
            match = regex.search(line.strip())
            if match:
                return match.group(1)
    return None

def extract_name_and_parents(lines):
    word_name, index_name, sim_name = find_best_text("NOME", lines)
    word_parents, index_parents, sim_parents = find_best_text("FILIAÇAO", lines)
    word_nat, index_nat, sim_nat = find_best_text("NATURALIDADE", lines)
    word_birth, index_birth, sim_birth = find_best_text("DATA DE NASCIMENTO", lines)

    if sim_name > 60:
        if extract_date([lines[index_name]]):
            index_name += 1
        if sim_parents > 60:
            name = ' '.join(lines[index_name:index_parents]).replace(word_name, '').strip()
            if sim_nat > 60:
                end_index = min(index_nat, index_birth) if sim_birth > 60 else index_nat
                parents = ' '.join(lines[index_parents:end_index]).replace(word_parents, '').strip()
                return name, parents
            return name, None
        return lines[index_name].replace(word_name, '').strip(), None
    return None, None

def extract_rg_t2(model, back_path, confidence_threshold=0.5, show_image=False, debug=False):
    try:
        lines, meta_data = pipeline_ocr(model, back_path, confidence_threshold, show_image, debug)

        rg = extract_rg_number(lines)
        cpf = extract_cpf(lines)
        birth_date = extract_date(lines)
        issue_date = extract_date(lines, start=3)
        name, parents = extract_name_and_parents(lines)

    except FileNotFoundError:
        print(f"[ERROR] File not found: {back_path}")
        return None, None

    extracted_data = {
        "Número do RG": rg,
        "CPF": cpf,
        "Nome": name,
        "Filiação": parents,
        "Data de expedição": issue_date,        
        "Data de nascimento": birth_date
    }

    return extracted_data, meta_data
