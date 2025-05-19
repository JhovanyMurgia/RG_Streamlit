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
                if -4 < angle < 4:
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
        cv2.imwrite('rotated_image.png', rotated_image)

        image_path = "rotated_image.png"

        result = run_ocr(model, image_path, show_image)

        # Apagar o arquivo temporário
        os.remove("rotated_image.png")

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
    return menor_distancia_global, melhor_palavra_global, indice_melhor_palavra

def extract_name(ocr_output):

    # Substituindo  / por espaço para facilitar a busca
    lista_limpa = [elemento.replace("/", " ") for elemento in ocr_output]

    dist ,word, line = encontrar_palavra_mais_proxima("NOME", lista_limpa)
    dist1 ,word1, line1 = encontrar_palavra_mais_proxima("NAME", lista_limpa)
    if dist > dist1:
        line = line1 
    
    nome = ocr_output[line + 1]
    nome = re.sub(r'[.,-/]', '', nome)

    return nome.strip()
  


#Fumção para obter filiação do RG
def extract_filiation(ocr_output):
    
    # Substituindo  / por espaço para facilitar a busca
    lista_limpa = [elemento.replace("/", " ") for elemento in ocr_output]
    
    dist1, word1, line1 = encontrar_palavra_mais_proxima("FILIAÇAO", lista_limpa)
    dist3, word3, line3 = encontrar_palavra_mais_proxima("FILIATION", lista_limpa)
    if dist1 > dist3:
        line1 = line3

    dist2, word2, line2 = encontrar_palavra_mais_proxima("EXPEDIDOR", lista_limpa)
    dist4, word4, line4 = encontrar_palavra_mais_proxima("CARD", lista_limpa)
    if dist2 > dist4:
        line2 = line4

    if line1 + 1 < len(ocr_output) and line2 < len(ocr_output):
        filiacao = "  ".join(ocr_output[line1 + 1:line2])
        # ajustando o retorno para exibir nome da mae e nome do pai
        #filiacao = re.sub(r'\b[a-zA-Z]*\d{2,12}[a-zA-Z]*\b|\b\d{2,12}\b|[.\-/]', '', filiacao)
        filiacao = re.sub(r'\b[a-zA-Z]*\d{2,12}[a-zA-Z]*\b|\b\d{1,12}\b|\b[a-zA-Z]\b|[.\-/]', '', filiacao)

        filiacao = filiacao.strip()
        filiacao = filiacao.replace("    ", "  ")
        filiacao = filiacao.replace("  ", " E ")
        filiacao = filiacao.replace(" E  E ", " E ")
        
        return filiacao.strip()
   

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
                return match.group(1)                    

    return None


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
def extract_rg_t3(model, path_frente, path_verso, limiar_conf=0.5, show_image=False, debug=False):
    try:
        result, meta_data_f = pipeline_ocr(model, path_frente, limiar_conf,
                              show_image=show_image, debug=debug)
        nome = extract_name(result)
        cpf = extract_cpf(result)         
        dt_nasc = extract_date(result)
    except FileNotFoundError as e:
        print(
            f"Imagem não fornecida ou não encontrada no caminho fornecido: {path_frente}")
        nome = None
        cpf = None
        dt_nasc = None

    try:
        result_v, meta_data_v = pipeline_ocr(
            model, path_verso, limiar_conf, show_image=show_image, debug=debug)
        
        filiacao = extract_filiation(result_v)                 
        dt_expedicao = extract_date(result_v)
    except FileNotFoundError as e:
        print(
            f"Imagem não fornecida ou não encontrada no caminho fornecido: {path_verso}")
        filiacao = None
        dt_expedicao = None

    dados = {
        "Data de Expedicao": dt_expedicao,
        "Nome": nome,
        "Filiacao": filiacao,
        "CPF": cpf,
        "Data de Nascimento": dt_nasc
    }
    return dados, meta_data_f, meta_data_v
