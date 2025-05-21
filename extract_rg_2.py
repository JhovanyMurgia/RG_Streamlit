import re
import cv2
import os
from rapidfuzz.distance import Levenshtein
import numpy as np


from auxiliary_functions import calculate_base_angle, average_angles_boxes, rotate_image, group_words_by_lines
from config_run_model import run_ocr


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
    lines = group_words_by_lines(words_data, tolerance=0.02)

    if debug:
        # Exibe as linhas como texto (opcional)
        for i, line_text in enumerate(lines):
            print(f"Linha {i + 1}: {line_text}")

    return lines, meta_data


# Função para encontrar o trecho mais próximo de um alvo em uma lista de textos resultante do OCR
def encontrar_trecho_mais_proximo(alvo, lista):
    melhor_trecho_global = None
    menor_distancia_global = float('inf')
    indice_melhor_trecho = -1
    similaridade_percentual = 0.0

    alvo_palavras = alvo.split()
    tamanho_alvo = len(alvo_palavras)  # número de palavras do alvo

    for i, texto in enumerate(lista):
        palavras = texto.split()
        melhor_trecho_local = None
        menor_distancia_local = float('inf')

        # Gerar n-gramas (sequências) do mesmo tamanho do alvo
        for j in range(len(palavras) - tamanho_alvo + 1):
            trecho = ' '.join(palavras[j:j + tamanho_alvo])

            distancia = Levenshtein.distance(alvo.upper(), trecho.upper())
            if distancia < menor_distancia_local:
                menor_distancia_local = distancia
                melhor_trecho_local = trecho

        if menor_distancia_local < menor_distancia_global:
            menor_distancia_global = menor_distancia_local
            melhor_trecho_global = melhor_trecho_local
            indice_melhor_trecho = i
            # Cálculo da similaridade percentual
            maior_len = max(len(alvo), len(melhor_trecho_local))
            similaridade_percentual = (
                1 - menor_distancia_local / maior_len) * 100

    return melhor_trecho_global, indice_melhor_trecho, similaridade_percentual


def extract_num_rg_antigo(ocr_output):

    rg_patterns = [
        # Após "REGISTRO GERAL"
        re.compile(r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})'),
        # (22.875.151-94 ou 22.875.151-9)
        re.compile(r'\b([0-9]{2,3}\.[0-9]{3}\.[0-9]{3}-[0-9]{1,2})\b'),
        # Formato com pontos (1.234.567 ou 12.345.678)
        re.compile(r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b'),
        # Formato com espaços (1 234 567 ou 12 345 678)
        re.compile(r'\b([0-9]{1,2}\s[0-9]{3}\s[0-9]{3})\b'),
        re.compile(r'\b([0-9]{3}\.[0-9]{3})\b')  # (123.456)

    ]

    for pattern in rg_patterns:
        for line in ocr_output[:4]:  # Limita a iteração às 4 primeiras linhas
            line = line.strip()
            match = pattern.search(line)
            if match:
                return match.group(1)  # Retorna o primeiro RG encontrado

    return None  # Retorna None se nenhum RG for encontrado


def extract_cpf_antigo(ocr_output):
    pattern = re.compile(r'\b([0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2})\b')

    for line in ocr_output[3:]: 
        line = line.strip()
        match = pattern.search(line)
        if match:
            return match.group(1)  # Retorna o primeiro CPF encontrado

    return None  # Retorna None se nenhum CPF for encontrado


def extract_dt_nasc_antigo(ocr_output):
    dt_nasc_patterns = [
        # Formato com barras (12/34/5678)
        re.compile(r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b'),
        # Formato com hífens (12-34-5678)
        re.compile(r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b'),
        # Formato com pontos (12.34.5678)
        re.compile(r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b'),
        
    ]

    for pattern in dt_nasc_patterns:
        for line in ocr_output:
            line = line.strip()
            match = pattern.search(line)
            if match:
                return match.group(1)

    return None


def extract_dt_expedicao_antigo(ocr_output):
    dt_exp_patterns = [
        # Formato com barras (12/34/5678)
        re.compile(r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b'),
        # Formato com hífens (12-34-5678)
        re.compile(r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b'),
        # Formato com pontos (12.34.5678)
        re.compile(r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b')
    ]

    for pattern in dt_exp_patterns:
        for line in ocr_output[3:]:  # Começa a partir da quinta linha
            line = line.strip()
            match = pattern.search(line)
            if match:
                return match.group(1)

    return None

# Fumção para obter nome e filiação do RG


def extract_name_filiation(ocr_output):
    word, line, perc = encontrar_trecho_mais_proximo("NOME", ocr_output)
    word1, line1, perc1 = encontrar_trecho_mais_proximo("FILIAÇAO", ocr_output)
    _, line2, perc2 = encontrar_trecho_mais_proximo(
        "NATURALIDADE", ocr_output)    
    _, line3, perc3 = encontrar_trecho_mais_proximo(
        "DATA DE NASCIMENTO", ocr_output)
    

    # Verificar se Nome tem mais de 60% de similaridade
    if perc > 60:
        # Verificar se tem uma data na msm linha do Nome
        dt_nasc_patterns = [
            # Formato com barras (12/34/5678)
            re.compile(r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b'),
            # Formato com hífens (12-34-5678)
            re.compile(r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b'),
            # Formato com pontos (12.34.5678)
            re.compile(r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b')
        ]
        data = None
        for pattern in dt_nasc_patterns:
            content = ocr_output[line].strip()
            match = pattern.search(content)
            if match:
                data = match.group(1)

        if data is not None:
            line = line + 1
        # Verifica se Filiacao tem mais de 60% de similaridade
        if perc1 > 60:
            nome = ' '.join(ocr_output[line:line1])
            nome = nome.replace(word, "")

            # Verifica se Naturalidade tem mais de 60% de similaridade
            if perc2 > 60:
                if perc3 > 60 and line2 > line3:
                    line2 = line3
                filiacao = ' '.join(ocr_output[line1:line2])
                filiacao = filiacao.replace(word1, "")
                return nome.strip(), filiacao.strip()
        else:
            nome = ocr_output[line]
            nome = nome.replace(word, "")
            return nome.strip(), None

    return None, None  

# Funcoes para fazer OCR de RGs com baixa qualidade


def extrair_texto_preto_sem_verde_melhorado(imagem_path, salvar_como=None):
    imagem = cv2.imread(imagem_path)

    if imagem is None:
        raise FileNotFoundError(
            f"Não foi possível abrir a imagem: {imagem_path}")

    # Medir brilho médio da imagem original
    gray_base = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(gray_base)

    # Definir fator de contraste com base no brilho
    if brilho_medio < 80:
        fator_contraste = 1.5
    elif brilho_medio < 130:
        fator_contraste = 2.0
    else:
        fator_contraste = 2.5

    # Aumentar contraste com normalização adaptativa
    imagem_float = imagem.astype(np.float32) / 255.0
    imagem_float = np.clip((imagem_float - 0.5) * fator_contraste + 0.5, 0, 1)
    imagem_contraste = (imagem_float * 255).astype(np.uint8)

    # Remover tons verdes
    b, g, r = cv2.split(imagem_contraste)
    mask_verde = (g > r + 30) & (g > b + 30)
    imagem_contraste[mask_verde] = [255, 255, 255]

    # Aumentar resolução (upscaling)
    imagem_contraste = cv2.resize(
        imagem_contraste, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Aplicar nitidez
    kernel_nitidez = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    imagem_contraste = cv2.filter2D(imagem_contraste, -1, kernel_nitidez)

    # Converter para escala de cinza e binarizar (texto preto, fundo branco)
    gray = cv2.cvtColor(imagem_contraste, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    # Aplicar morfologia para remover pequenos ruídos sem apagar detalhes
    kernel = np.ones((2, 1), np.uint8)
    final = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Salvar imagem final se desejado
    if salvar_como:
        cv2.imwrite(salvar_como, final)


    return final

def extract_rg_preprocess(ocr_output):
    rg = None
    dt_nasc = None
    dt_exp = None
    nome = None
    filiacao = None

    rg_patterns = [
        # Após "REGISTRO GERAL"
        re.compile(r'REGISTRO GERAL[\s:]*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3})'),
        # (22.875.151-94 ou 22.875.151-9)
        re.compile(r'\b([0-9]{2,3}\.[0-9]{3}\.[0-9]{3}-[0-9]{1,2})\b'),
        # Formato com pontos (1.234.567 ou 12.345.678)
        re.compile(r'\b([0-9]{1,2}\.[0-9]{3}\.[0-9]{3})\b'),
        # Formato com espaços (1 234 567 ou 12 345 678)
        re.compile(r'\b([0-9]{1,2}\s[0-9]{3}\s[0-9]{3})\b'),
        # (123.456)
        re.compile(r'\b([0-9]{3}\.[0-9]{3})\b')  
    ]

    dt_patterns = [
            # Formato com barras (12/34/5678)
            re.compile(r'\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b'),
            # Formato com hífens (12-34-5678)
            re.compile(r'\b([0-9]{2}-[0-9]{2}-[0-9]{4})\b'),
            # Formato com pontos (12.34.5678)
            re.compile(r'\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b')
    ]

    for pattern in rg_patterns:
        i = 0        
        for line in ocr_output[:4]:  # Limita a iteração às 4 primeiras linhas
            line = line.strip()
            match = pattern.search(line)
            if match:
                rg = match.group(1)  # Retorna o primeiro RG encontrado
                line_rg = i
                break
            i += 1
        if rg is not None:
            break

    for pattern in dt_patterns:
        i = 0
        for line in ocr_output:
            line = line.strip()
            match = pattern.search(line)
            if match:
                dt_nasc = match.group(1)
                line_dt_nasc = i
                break
            i += 1
        if dt_nasc is not None:
            break
    
    if dt_nasc is not None:
        for pattern in dt_patterns:
            i = 0
            for line in ocr_output[line_dt_nasc + 1:]:
                line = line.strip()
                match = pattern.search(line)
                if match:
                    dt_exp = match.group(1)
                    line_dt_exp = i + line_dt_nasc + 1
                    break
                i += 1
            if dt_exp is not None:
                break
    
    if rg is not None:
        line_nome = line_rg + 1
        if dt_nasc is not None:
            if line_rg < line_dt_nasc:
                line_nome = line_dt_nasc + 1

            nome = ocr_output[line_nome]
            if dt_exp is not None:
                filiacao = ' '.join(ocr_output[line_nome + 1:line_dt_exp])
            else:
                filiacao = ' '.join(ocr_output[line_nome + 1:line_nome + 3])
            
        else:
            nome = ocr_output[line_nome]
            filiacao = ' '.join(ocr_output[line_nome + 1:line_nome + 3])
    
    else:
        nome = ocr_output[2]
        filiacao = ' '.join(ocr_output[3:5])

    return rg, nome, filiacao, dt_nasc, dt_exp
        


# Pipeline para extrair informações do RG antigo
def extract_rg_t2(model, path_verso, limiar_conf=0.5, show_image=False, debug=False):
    try:
        # Chama a função e obtém o resultado
        lines, meta_data = pipeline_ocr(
            model, path_verso, limiar_conf=limiar_conf, show_image=show_image, debug=debug)
        # Extrai as informações
        rg = extract_num_rg_antigo(lines)
        nome, filiacao = extract_name_filiation(lines)
        cpf = extract_cpf_antigo(lines)
        dt_nasc = extract_dt_nasc_antigo(lines)
        dt_exp = extract_dt_expedicao_antigo(lines)

        if rg == None or nome == None:
            # Chama a função de pré-processamento
            extrair_texto_preto_sem_verde_melhorado(
                imagem_path=path_verso,
                salvar_como="saida_processada.png",  # ou None, se não quiser salvar
                mostrar_etapas=False  # veja o passo a passo
                )
            imagem_melhorada = "saida_processada.png"

            lines, meta_data = pipeline_ocr(
                model, imagem_melhorada, limiar_conf=limiar_conf, show_image=show_image, debug=debug)
            rg2, nome2, filiacao2, dt_nasc2, dt_exp2 = extract_rg_preprocess(lines)
            if rg == None:
                rg = rg2
            if nome == None:
                nome = nome2
            if filiacao == None:
                filiacao = filiacao2
            if dt_nasc == None:
                dt_nasc = dt_nasc2
            if dt_exp == None:
                dt_exp = dt_exp2 
            os.remove("saida_processada.png")     

    except FileNotFoundError as e:
        print(
            f"Imagem não fornecida ou não encontrada no caminho fornecido: {path_verso}")
        nome = None
        filiacao = None
        dt_nasc = None
        rg = None
        cpf = None
        dt_exp = None

    dados = {
        "RG": rg,
        "Nome": nome,
        "Filiacao": filiacao,
        "CPF": cpf,
        "Data de Nascimento": dt_nasc,
        "Data de Expedicao": dt_exp
    }
    
    return dados, meta_data
