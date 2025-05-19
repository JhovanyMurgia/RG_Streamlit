import re
import cv2
import os

from rapidfuzz.distance import Levenshtein

from auxiliary_functions import calculate_base_angle, average_angles_boxes, rotate_image, group_words_by_lines
from config_run_model import run_ocr




# Funções utilizadas para extrair informações do texto de ambos os tipos de RG
#############################################################################

# Pipeline para OCR
def pipeline_ocr(model, image_path, limiar_conf=0.5, show_image=False, debug=False):

    result = run_ocr(model, image_path, show_image)
    meta_data = result

    # Obter a inclinação dos retângulos 
    angle_list = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                vertices = word.geometry
                angle = calculate_base_angle(vertices)
                # Excluir valores discrepantes
                if -10 < angle < 10:
                    angle_list.append(angle)
    
    if len(angle_list) > 0:
        # Função que ordena e calcula a média dos quatro ângulos centrais da lista
        mean_angle = average_angles_boxes(angle_list)
    else:
        mean_angle = 0
    

    if mean_angle > 1 or mean_angle < -1:
        # Ajusta inclinação da imagem
        rotated_image = rotate_image(image_path, mean_angle)

        # Salvar a imagem rotacionada
        cv2.imwrite('rotated_image.jpg', rotated_image)

        image_path = "rotated_image.jpg"

        result = run_ocr(model, image_path, show_image)

        # Apagar o arquivo temporário
        os.remove("rotated_image.jpg")

    # Extrair o texto das linhas dentro do intervalo ajustado
    words_data = []

    # Extrair o texto e a geometria
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > limiar_conf:
                    word_data = {
                        "text": word.value,  # O texto da palavra
                        "geometry": word.geometry  # Coordenadas normalizadas
                    }
                    words_data.append(word_data)

    # Agrupa as palavras em linhas
    lines = group_words_by_lines(words_data)

    if debug:
        # Exibe as linhas como texto (opcional)
        for i, line_text in enumerate(lines):
            print(f"Linha {i + 1}: {line_text}")

    return lines, meta_data



# Funções utilizadas para extrair informações do texto do novo RG
################################################################

# Função para encontrar a palavra mais semelhante à palavra alvo
def encontrar_palavra_mais_proxima(alvo, lista):
    melhor_palavra_global = None
    menor_distancia_global = float('inf')
    indice_melhor_palavra = -1

    for i, texto in enumerate(lista):
        # Separar texto em palavras
        palavras = texto.split()

        # Inicializar variáveis de controle para cada elemento
        melhor_palavra_local = None
        menor_distancia_local = float('inf')

        # Comparar cada palavra do texto com o alvo
        for palavra in palavras:
            distancia = Levenshtein.distance(alvo.upper(), palavra.upper())
            if distancia < menor_distancia_local:
                menor_distancia_local = distancia
                melhor_palavra_local = palavra

        # Atualizar variáveis globais se encontrar uma palavra mais próxima
        if menor_distancia_local < menor_distancia_global:
            menor_distancia_global = menor_distancia_local
            melhor_palavra_global = melhor_palavra_local
            indice_melhor_palavra = i

    # Resultado final
    return melhor_palavra_global, indice_melhor_palavra


#Fumção para obter nome e filiação do RG
def extract_name_filiation(ocr_output):
    word, line = encontrar_palavra_mais_proxima("NOME", ocr_output)
    word1, line1 = encontrar_palavra_mais_proxima("FILIAÇAO", ocr_output)
    word2, line2 = encontrar_palavra_mais_proxima("DATA", ocr_output)

    nome = ' '.join(ocr_output[line:line1])
    nome = nome.replace(word, "")

    filiacao = ' '.join(ocr_output[line1:line2])
    filiacao = filiacao.replace(word1, "")

    return nome.strip(), filiacao.strip()

#Função para obter o número do RG
def extract_num_rg_novo(ocr_output):
    rg_patterns = [
        re.compile(r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})'),
        re.compile(r'\b([0-9]{2,3}\.[0-9]{3}\.[0-9]{3}-[0-9]{1,2})\b'),
        re.compile(r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b'),
        re.compile(r'\b([0-9]{1,2}\s[0-9]{3}\s[0-9]{3})\b'),
        re.compile(r'\b([0-9]{3}\.[0-9]{3})\b')
    ]

    for pattern in rg_patterns:
        for line in ocr_output:
            line = line.strip()
            match = pattern.search(line)
            if match:
                return match.group(1)

    return None


#Função para obter o número do CPF
def extract_cpf(ocr_output):
    cpf_patterns = [
        re.compile(r'CPF[\s:]*([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})'),
        re.compile(r'\b([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})\b')
    ]

    for i, line in enumerate(ocr_output):
        line = line.strip()
        for pattern in cpf_patterns:
            match = pattern.search(line)
            if match:
                cpf = match.group(1)
                # Busca o RG nas linhas abaixo do CPF
                rg = extract_num_rg_novo(ocr_output[i+1:])
                if rg:
                    return cpf, rg
                else:
                    return cpf, None

    return None, None 


#Função para obter a data de nascimento e expedição
def extract_date(ocr_output):

    dt_patterns = [
        # Formato com barras (12/34/5678)
        re.compile(r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b'),
        # Formato com hífens (12-34-5678)
        re.compile(r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b'),
        # Formato com pontos (12.34.5678)
        re.compile(r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b')
    ]

    for pattern in dt_patterns:
        for line in ocr_output:  
            line = line.strip()
            match = pattern.search(line)
            if match:
                return match.group(1)  

    return None  


# Pipeline para extrair informações do RG novo
def extract_rg_t1(model, path_frente, path_verso, limiar_conf=0.5, show_image=False, debug=False):
    try:
        result, meta_data_f = pipeline_ocr(model, path_frente, limiar_conf,
                              show_image=show_image, debug=debug)
        nome, filiacao = extract_name_filiation(result)
        dt_nasc = extract_date(result)
    except FileNotFoundError as e:
        print(
            f"Imagem não fornecida ou não encontrada no caminho fornecido: {path_frente}")
        nome = None
        filiacao = None
        dt_nasc = None

    try:
        result_v, meta_data_v = pipeline_ocr(
            model, path_verso, limiar_conf, show_image=show_image, debug=debug)
        cpf, rg = extract_cpf(result_v)
        if rg is None:
            rg = extract_num_rg_novo(result_v)
        dt_expedicao = extract_date(result_v)
    except FileNotFoundError as e:
        print(
            f"Imagem não fornecida ou não encontrada no caminho fornecido: {path_verso}")
        cpf = None
        rg = None
        dt_expedicao = None

    dados = {
        "RG": rg,
        "Data de Expedicao": dt_expedicao,
        "Nome": nome,
        "Filiacao": filiacao,
        "CPF": cpf,
        "Data de Nascimento": dt_nasc
    }
    return dados, meta_data_f, meta_data_v
