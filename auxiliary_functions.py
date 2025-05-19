import math
import cv2


def calculate_base_angle(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]

    # Calcular a diferença nas coordenadas x e y
    dx = x2 - x1
    dy = y2 - y1

    # Calcular o ângulo em radianos
    angle_rad = math.atan2(dy, dx)

    # Converter o ângulo para graus
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        return angle_deg + 0.5
    elif angle_deg > 0:
        return angle_deg - 0.5
    else:
        return angle_deg

def average_angles_boxes(angle_list):
    # Ordena a lista
    sorted_list = sorted(angle_list)

    # Calcula o índice central
    n = len(sorted_list)
    middle = n // 2

    # Verifica se a lista tem pelo menos 4 elementos
    if n < 4:
        return sum(angle_list) / len(angle_list)

    # Se o número de elementos for ímpar, pega os 4 valores centrais ao redor do meio
    if n % 2 == 1:
        central_values = sorted_list[middle-1:middle+3]
    else:
        # Se o número de elementos for par, pega os 4 valores centrais ao redor do meio
        central_values = sorted_list[middle-2:middle+2]

    # Calcula a média dos 4 valores centrais
    mean_angle = sum(central_values) / 4

    return mean_angle

def rotate_image(image_path, angle):
    # Carregar a imagem
    image = cv2.imread(image_path)

    # Obter as dimensões da imagem
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calcular a matriz de rotação
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotacionar a imagem usando interpolação de alta qualidade
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LANCZOS4)

    return rotated_image

# Funções para organizar o OCR por linha
def extract_y_center(geometry):
    top_y = min(coord[1] for coord in geometry)
    bottom_y = max(coord[1] for coord in geometry)
    return (top_y + bottom_y) / 2

def group_words_by_lines(words_data, tolerance=0.01):
    # Adiciona o centro Y para cada palavra
    for word in words_data:
        word["y_center"] = extract_y_center(word["geometry"])

    # Ordena as palavras pelo centro Y (de cima para baixo)
    words_data.sort(key=lambda w: w["y_center"])

    lines = []
    current_line = []
    last_y = None

    for word in words_data:
        if last_y is None or abs(word["y_center"] - last_y) <= tolerance:
            # Adiciona à linha atual
            current_line.append(word)
        else:
            # Nova linha detectada
            lines.append(current_line)
            current_line = [word]
        last_y = word["y_center"]

    # Adiciona a última linha se necessário
    if current_line:
        lines.append(current_line)

    # Ordena as palavras de cada linha pelo eixo X (esquerda para a direita)
    for line in lines:
        line.sort(key=lambda w: min(coord[0] for coord in w["geometry"]))

    # Converte as linhas agrupadas em uma lista de strings
    list_lines = [' '.join(word['text'] for word in line) for line in lines]

    return list_lines


    
