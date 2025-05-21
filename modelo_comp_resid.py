from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
import requests
import time
import os

def extrair_ceps_de_texto(texto):
    """
    Extrai CEPs do texto usando expressão regular específica para CEPs do ES (29xxx-xxx).
    Tolera espaços e outros caracteres que o OCR pode inserir erroneamente.
    """
    # Padrão para CEPs do ES com tolerância a erros de OCR
    padrao_cep_es = r"29\s*\d\s*\d\s*\d\s*[-\s]?\s*\d\s*\d\s*\d"
    
    # Encontra todos os CEPs
    ceps_brutos = re.findall(padrao_cep_es, texto)
    
    # Limpa espaços e caracteres não numéricos, exceto o hífen entre 5º e 6º dígitos
    ceps_formatados = []
    for cep in ceps_brutos:
        # Remove todos os espaços
        cep_limpo = re.sub(r"\s+", "", cep)
        
        # Verifica se já tem hífen, senão adiciona
        if '-' not in cep_limpo and len(cep_limpo) >= 8:
            cep_limpo = cep_limpo[:5] + '-' + cep_limpo[5:8]
        
        # Verifica se está no formato correto antes de adicionar
        if re.match(r"29\d{3}-\d{3}", cep_limpo):
            ceps_formatados.append(cep_limpo)
    
    return ceps_formatados

def consultar_viacep(cep):
    """
    Consulta a API ViaCEP para obter informações sobre o CEP.
    Retorna um dicionário com as informações ou None em caso de erro.
    """
    # Remove caracteres não numéricos do CEP para a consulta
    cep_numerico = re.sub(r"\D", "", cep)
    
    # URL da API ViaCEP
    url = f"https://viacep.com.br/ws/{cep_numerico}/json/"
    
    try:
        # Faz a requisição para a API
        response = requests.get(url, timeout=5)
        
        # Verifica se a requisição foi bem-sucedida
        if response.status_code == 200:
            dados = response.json()
            
            # Verifica se o CEP foi encontrado (a API retorna "erro": true quando não encontra)
            if "erro" not in dados:
                return dados
        
        # Aguarda um pequeno intervalo para não sobrecarregar a API
        time.sleep(0.5)
        return None
    
    except Exception as e:
        print(f"Erro ao consultar o CEP {cep}: {str(e)}")
        return None

def processar_pdf(caminho_arquivo, modelo_ocr=None):
    """
    Processa um único arquivo PDF e retorna um dicionário com os resultados da extração de CEP.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo PDF a ser processado
        modelo_ocr (doctr.models.ocr.OCRPredictor, optional): Modelo OCR pré-carregado. 
                                                            Se None, será carregado um novo modelo.
    
    Returns:
        dict: Dicionário com os resultados da extração, incluindo:
              - sucesso (bool): Se foi encontrado pelo menos um CEP válido
              - nome_arquivo (str): Nome do arquivo processado
              - ceps_encontrados (list): Lista de todos os CEPs encontrados no texto
              - quantidade_ceps (int): Quantidade de CEPs encontrados
              - detalhes_cep (dict or None): Detalhes do primeiro CEP válido encontrado via ViaCEP ou None
    """
    # Se não foi fornecido um modelo OCR, carrega um novo
    if modelo_ocr is None:
        print("Carregando o modelo OCR...")
        modelo_ocr = ocr_predictor(pretrained=True)
    
    try:
        # Lista de CEPs de empresas conhecidas para filtrar
        lista_empresas_cep = ["29050-335", "29050-310", "29931-910", "29162-206", "29165-827"]
        
        # Extrai o nome do arquivo do caminho
        nome_arquivo = os.path.basename(caminho_arquivo)
        
        # Carrega o documento PDF
        doc = DocumentFile.from_pdf(caminho_arquivo)
        
        # Realiza OCR
        resultado = modelo_ocr(doc)
        dados_exportados = resultado.export()
        
        # Junta todo o texto do OCR
        texto_completo = ""
        for pagina in dados_exportados['pages']:
            for bloco in pagina['blocks']:
                for linha in bloco['lines']:
                    for palavra in linha['words']:
                        texto_completo += palavra['value'] + " "  # Espaço entre palavras
        
        # Extrai CEPs do texto
        ceps_encontrados = extrair_ceps_de_texto(texto_completo)
        
        # Filtra CEPs (remove duplicados e CEPs de empresas)
        lista_cep_filtrado = []
        for cep in ceps_encontrados:
            if cep not in lista_empresas_cep and cep not in lista_cep_filtrado:
                lista_cep_filtrado.append(cep)
        
        # Procura pelo primeiro CEP válido
        detalhe_cep = None
        for cep in lista_cep_filtrado:
            dados_cep = consultar_viacep(cep)
            if dados_cep and len(cep) == 9:  # Verifica se o CEP tem 9 caracteres (incluindo o hífen)
                detalhe_cep = dados_cep
                break
        
        # Prepara o resultado
        resultado = detalhe_cep
        
        return resultado
        
    except Exception as e:
        print(f"Erro ao processar {caminho_arquivo}: {str(e)}")
        return {
            "sucesso": False,
            "nome_arquivo": os.path.basename(caminho_arquivo),
            "ceps_encontrados": [],
            "quantidade_ceps": 0,
            "detalhes_cep": None,
            "erro": str(e)
        }



    
 