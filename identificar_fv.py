import cv2
from rapidfuzz.distance import Levenshtein
import re
import os
import numpy as np
import cv2
import fitz
from ultralytics import YOLO

from config_run_model import run_ocr

from extract_rg_1 import extract_rg_t1
from extract_rg_2 import extract_rg_t2
from extract_rg_3 import extract_rg_t3

def carregar_yolo():
    model_path = "./weights/best.pt"
    model = YOLO(model_path)
    return model


############################################
######### Recortar imagens #################


def detect_rg(caminho_pdf, model):
    

    # Lista para armazenar imagens recortadas
    recortes = []

    # Abre o PDF
    doc = fitz.open(caminho_pdf)

    for pagina_idx, pagina in enumerate(doc):
        pix = pagina.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array


        # salvar o RGB do pdf carregado
        #pdf = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #cv2.imwrite(f"imagem_pdf_{pagina_idx + 1}.png", pdf)


        # Detecta objetos com YOLO
        results = model(img)

        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cropped = img[y1:y2, x1:x2]
                recortes.append(cropped)
        else:
            print(f"Nenhuma detecção na página {pagina_idx + 1}")

    doc.close()
    return recortes



###########################################
########### Dividir RG ####################

def dividir_rg(imagens):  

    if len(imagens) == 0:
        print("[ERRO] Nenhuma imagem válida encontrada na pasta.")
    elif len(imagens) == 1:
        imagem = imagens[0]
        altura, largura = imagem.shape[:2]
        
        if altura > largura:
            if altura >= 2*largura:
                
                # Rotaciona 90 graus para a direita
                imagem_rotacionada = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)

                # Dividir a imagem em duas metades
                altura, largura = imagem_rotacionada.shape[:2]
                metade = largura // 2

                esquerda = imagem_rotacionada[:, :metade]
                direita = imagem_rotacionada[:, metade:]             

                # Salvar as imagens como img1 e img2 e apagar a imagem origonal
                nome_arquivo1 = "recortes/img_1.png"
                cv2.imwrite(nome_arquivo1, direita)
                nome_arquivo2 = "recortes/img_2.png"
                cv2.imwrite(nome_arquivo2, esquerda)
            
            else:
                altura, largura = imagem.shape[:2]
                metade = altura // 2

                superior = imagem[:metade, :]     # Da linha 0 até a metade da altura
                inferior = imagem[metade:, :]   # Da metade da altura até o final

                # Salvar as imagens como img1 e img2 e apagar a imagem origonal
                nome_arquivo1 = "recortes/img_1.png"
                cv2.imwrite(nome_arquivo1, superior)
                nome_arquivo2 = "recortes/img_2.png"
                cv2.imwrite(nome_arquivo2, inferior)
            
        else:   
            # Dividir a imagem em duas metades
            altura, largura = imagem.shape[:2]
            metade = largura // 2

            esquerda = imagem[:, :metade]
            direita = imagem[:, metade:]

            # Salvar as imagens como img1 e img2 e apagar a imagem origonal
            nome_arquivo1 = "recortes/img_1.png"
            cv2.imwrite(nome_arquivo1, direita)
            nome_arquivo2 = "recortes/img_2.png"
            cv2.imwrite(nome_arquivo2, esquerda)  
  
    else:
        direita = imagens[0]
        esquerda = imagens[1]
        # Salvar as imagens como img1 e img2 e apagar a imagem origonal
        nome_arquivo1 = "recortes/img_1.png"
        cv2.imwrite(nome_arquivo1, direita)
        nome_arquivo2 = "recortes/img_2.png"
        cv2.imwrite(nome_arquivo2, esquerda) 



def encontrar_palavra_mais_proxima(alvo, texto):
    palavras = texto.split()
    melhor_palavra = None
    menor_distancia = float('inf')
    similaridade_percentual = 0.0

    for palavra in palavras:
        distancia = Levenshtein.distance(alvo.upper(), palavra.upper())
        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_palavra = palavra
            maior_len = max(len(alvo), len(palavra))
            similaridade_percentual = (1 - distancia / maior_len) * 100

    return melhor_palavra, similaridade_percentual


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
            _, perc1 = encontrar_palavra_mais_proxima("NOME", texto2)
            _, perc2 = encontrar_palavra_mais_proxima(
                "FILIAÇAO", texto2)

            if perc1 > 85 or perc2 > 70:
                print("RG T1")
                dados, meta_data_f, meta_data_v = extract_rg_t1(
                    model, frente, verso, limiar_conf=0, show_image=False, debug=False)
                print(dados)
                return dados, meta_data_f, meta_data_v

            else:
                print("RG T2")
                dados, meta_data_v = extract_rg_t2(
                    model, verso, limiar_conf=0, show_image=False, debug=False)
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

                _, perc1 = encontrar_palavra_mais_proxima(
                    "NOME", texto1)
                _, perc2 = encontrar_palavra_mais_proxima(
                    "FILIAÇAO", texto1)

                if perc1 > 85 or perc2 > 70:
                    print("RG T1")
                    dados, meta_data_f, meta_data_v = extract_rg_t1(
                        model, frente, verso, limiar_conf=0, show_image=False, debug=False)
                    print(dados)
                    return dados, meta_data_f, meta_data_v
                else:
                    print("RG T2")
                    dados, meta_data_v = extract_rg_t2(
                        model, verso, limiar_conf=0, show_image=False, debug=False)
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
                model, frente, verso, limiar_conf=0, show_image=False, debug=False)
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
                model, frente, verso, limiar_conf=0, show_image=False, debug=False)
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
    dividir_rg(imagens)

      

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
