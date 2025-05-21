import os
import re
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


def load_ocr_model_cadunico():
    """
    Carrega o modelo OCR DocTR.
    """
    return ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True, assume_straight_pages=False)


def run_ocr(model, file_path, show_image=True):
    """
    Executa OCR no documento especificado (imagem ou PDF) e exibe a imagem com resultados.
    """
    if file_path.lower().endswith('.pdf'):
        doc = DocumentFile.from_pdf(file_path)
    else:
        doc = DocumentFile.from_images(file_path)

    result = model(doc)
    if show_image:
        result.show()
    return result


def extract_numbers(text):
    """
    Encontra todos os números com 8 ou mais dígitos em sequência no texto extraído.
    """
    return re.findall(r'\b\d{8,}\b', text)


def extract_names(text):
    """
    Extrai possíveis nomes próprios do texto considerando maiúsculas sem acento e ao menos duas palavras.
    """
    pattern = r'(?<=\n)\b([A-Z]{3,}(?:\s+[A-Z]{2,}){1,})\b'
    candidates = re.findall(pattern, text)
    blacklist = {
        "NIS", "PIS", "RUA", "FOLHA", "RESUMO", "CADASTRO", "UNICO", "RELATIVAS", "CEP",
        "CASA", "RF", "RENDA", "PER", "CAPITA", "FAMILIA", "PROXIMO", "ANTIGO", "FORMIGAO",
        "III", "RETIRO", "SAUDOSO", "NOVA", "VILA", "VELHA", "PLANALTO", "SERRANO",
        "BLOCO", "CACHOEIRO", "ITAPEMIRIM", "TRES", "MORADIAS", "MOXUARA", "SAO",
        "RESPONSAVEL", "FAMILIAR", "SANTA", "MARECHAL", "CASTELO", "PARQUE",
        "RESIDENCIAL", "MESTRE", "PERTO", "REGIAO", "NOVO", "MEXICO", "EDF",
        "VALE", "ENCANTADO", "OPERARIO", "AVENIDA", "RIO", "FLETRICISTAS", "SENNA"
    }
    valid_names = []
    for cand in candidates:
        words = cand.split()
        if not any(w in blacklist for w in words):
            valid_names.append(cand)
    return valid_names


def process_document(model, file_path, show_image=True):
    """
    Executa OCR no arquivo especificado pelo caminho completo e exibe resultados no notebook.
    Retorna um dicionário com texto, números e nomes extraídos.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    result = run_ocr(model, file_path, show_image=show_image)
    text = result.render()
    numbers = extract_numbers(text)
    names = extract_names(text)

    print(f"Arquivo processado: {os.path.basename(file_path)}")
    print("Números encontrados:", numbers if numbers else 'Nenhum')
    print("Nomes encontrados:", names if names else 'Nenhum')


    names.insert(0, "Código familiar")

    data =  {        
        'Nome': names,
        'Número': numbers
    }

    return data


