# pipeline/full_pipeline.py

from ultralytics import YOLO
from config.ocr_model_config import run_ocr
from ocr.extract_rg_type1 import extract_rg_t1
from ocr.extract_rg_type2 import extract_rg_t2
from ocr.extract_rg_type3 import extract_rg_t3

import fitz
import numpy as np
import cv2
import os
import re
from rapidfuzz.distance import Levenshtein


def load_yolo_model():
    model_path = "./weights/best.pt"
    return YOLO(model_path)


def detect_rg(pdf_path, model):
    cropped_images = []
    doc = fitz.open(pdf_path)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model(img)

        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cropped = img[y1:y2, x1:x2]
                cropped_images.append(cropped)
        else:
            print(f"[INFO] No detection on page {i+1}")

    doc.close()
    return cropped_images


def split_rg(images):
    os.makedirs("recortes", exist_ok=True)

    if len(images) == 0:
        print("[ERROR] No image detected")
        return

    elif len(images) == 1:
        image = images[0]
        h, w = image.shape[:2]

        if h > w and h >= 2 * w:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            h, w = image.shape[:2]
            mid = w // 2
            cv2.imwrite("recortes/img_1.png", image[:, mid:])
            cv2.imwrite("recortes/img_2.png", image[:, :mid])

        elif h > w:
            mid = h // 2
            cv2.imwrite("recortes/img_1.png", image[:mid, :])
            cv2.imwrite("recortes/img_2.png", image[mid:, :])

        else:
            mid = w // 2
            cv2.imwrite("recortes/img_1.png", image[:, mid:])
            cv2.imwrite("recortes/img_2.png", image[:, :mid])

    else:
        cv2.imwrite("recortes/img_1.png", images[0])
        cv2.imwrite("recortes/img_2.png", images[1])


def find_best_word(target, text):
    words = text.split()
    best_word = None
    best_score = float("inf")

    for word in words:
        distance = Levenshtein.distance(target.upper(), word.upper())
        if distance < best_score:
            best_score = distance
            best_word = word

    return best_word, (1 - best_score / max(len(target), len(best_word))) * 100


def etapa_final(arq1, arq2, model_classifier, model):
    img1 = cv2.imread(arq1)
    img2 = cv2.imread(arq2)

    altura, largura = img1.shape[:2]

    if largura < altura:
        # Rotaciona 90 graus para a direita
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(arq1, img1)

    altura, largura = img2.shape[:2]

    if largura < altura:
        # Rotaciona 90 graus para a direita
        img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(arq2, img2)

    result1 = run_ocr(model_classifier, arq1, show_image=False)
    media_confidencia1 = 0
    tot_palavras = 0
    texto1 = ""
    for block in result1.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > 0:
                    texto1 += word.value + " "
                    media_confidencia1 += word.confidence
                    tot_palavras += 1

    media_confidencia1 = media_confidencia1/tot_palavras

    result2 = run_ocr(model_classifier, arq2, show_image=False)
    texto2 = ""
    media_confidencia2 = 0
    tot_palavras = 0
    for block in result2.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > 0:
                    texto2 += word.value + " "
                    media_confidencia2 += word.confidence
                    tot_palavras += 1
    media_confidencia2 = media_confidencia2/tot_palavras

    if media_confidencia1 < 0.6:
        print("Ajustando orientação da imagem 1")
        img1 = cv2.rotate(img1, cv2.ROTATE_180)
        cv2.imwrite(arq1, img1)

        result1 = run_ocr(model_classifier, arq1, show_image=False)
        media_confidencia1 = 0
        tot_palavras = 0
        texto1 = ""
        for block in result1.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > 0:
                        texto1 += word.value + " "
                        media_confidencia1 += word.confidence
                        tot_palavras += 1

    if media_confidencia2 < 0.6:
        print("Ajustando orientação da imagem 2")
        img2 = cv2.rotate(img2, cv2.ROTATE_180)
        cv2.imwrite(arq2, img2)

        result2 = run_ocr(model_classifier, arq2, show_image=False)
        media_confidencia2 = 0
        tot_palavras = 0
        texto2 = ""
        for block in result2.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > 0:
                        texto2 += word.value + " "
                        media_confidencia2 += word.confidence
                        tot_palavras += 1

    # Lista de padrões para RG
    rg_patterns = [
        re.compile(r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})'),
        re.compile(r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b')
    ]

    verso = None
    frente = None

    for pattern in rg_patterns:
        if re.search(pattern, texto1):
            verso = arq1
            frente = arq2

            # Salvar a img da frente e do verso do rg
            cv2.imwrite("img_frente.png", img2)
            cv2.imwrite("img_verso.png", img1)

            # Buscar NOME e FILIAÇÃO
            _, perc1 = find_best_word("NOME", texto2)
            _, perc2 = find_best_word(
                "FILIAÇAO", texto2)

            if perc1 > 85 or perc2 > 70:
                print("RG T1")
                dados, meta_data_f, meta_data_v = extract_rg_t1(
                    model, frente, verso, confidence_threshold=0, show_image=False, debug=False)
                print(dados)
                return dados, meta_data_f, meta_data_v

            else:
                print("RG T2")
                dados, meta_data_v = extract_rg_t2(
                    model, verso, confidence_threshold=0, show_image=False, debug=False)
                print(dados)
                return dados, result2, meta_data_v
            break
    if verso is None:
        for pattern in rg_patterns:
            if re.search(pattern, texto2):
                verso = arq2
                frente = arq1

                # Salvar a img da frente e do verso do rg
                cv2.imwrite("img_frente.png", img1)
                cv2.imwrite("img_verso.png", img2)

                _, perc1 = find_best_word(
                    "NOME", texto1)
                _, perc2 = find_best_word(
                    "FILIAÇAO", texto1)

                if perc1 > 85 or perc2 > 70:
                    print("RG T1")
                    dados, meta_data_f, meta_data_v = extract_rg_t1(
                        model, frente, verso, confidence_threshold=0, show_image=False, debug=False)
                    print(dados)
                    return dados, meta_data_f, meta_data_v
                else:
                    print("RG T2")
                    dados, meta_data_v = extract_rg_t2(
                        model, verso, confidence_threshold=0, show_image=False, debug=False)
                    print(dados)
                    return dados, result1, meta_data_v
                break

    if verso is None:
        padrao = r'\b[0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2}\b'

        if re.search(padrao, texto1):
            frente = arq1
            verso = arq2

            # Salvar a img da frente e do verso do rg
            cv2.imwrite("img_frente.png", img1)
            cv2.imwrite("img_verso.png", img2)

            print("RG T3")
            dados, meta_data_f, meta_data_v = extract_rg_t3(
                model, frente, verso, confidence_threshold=0, show_image=False, debug=False)
            print(dados)
            return dados, meta_data_f, meta_data_v
        if re.search(padrao, texto2):
            frente = arq2
            verso = arq1

            # Salvar a img da frente e do verso do rg
            cv2.imwrite("img_frente.png", img2)
            cv2.imwrite("img_verso.png", verso)

            print("RG T3")
            dados, meta_data_f, meta_data_v = extract_rg_t3(
                model, frente, verso, confidence_threshold=0, show_image=False, debug=False)
            print(dados)
            return dados, meta_data_f, meta_data_v

    if verso is None:
        print("Tipo de RG não identificado")
        return None, None, None

    return None, None, None


def listar_nomes_arquivos(pasta):
    return [
        nome for nome in os.listdir(pasta)
        if os.path.isfile(os.path.join(pasta, nome))
    ]


def pipeline_completo_rg(caminho_pdf):
    # Detecta RG
    imagens = detect_rg(caminho_pdf)

    pasta_recortes = "./recortes"

    for arquivo in os.listdir(pasta_recortes):
        caminho_pdf = os.path.join(pasta_recortes, arquivo)
        os.remove(caminho_pdf)

    # Divide RG
    split_rg(imagens)

    arquivos = listar_nomes_arquivos(pasta_recortes)

    if len(arquivos) == 2:
        arq1 = os.path.join(pasta_recortes, arquivos[0])
        arq2 = os.path.join(pasta_recortes, arquivos[1])

        # Executa a etapa final
        dados, meta_data_f, meta_data_v = etapa_final(arq1, arq2)
        return dados, meta_data_f, meta_data_v

    else:
        print("Nenhum RG disponível para")
        return None, None, None
