import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
import torch
torch.classes.__path__ = []

from identificar_fv import dividir_rg, detect_rg, etapa_final
from modelo_cadunico import load_ocr_model, process_document
from modelo_comp_resid import processar_pdf


def desenhar_bounding_boxes(uploaded_file, result):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    annotated_image = image_np.copy()
    h, w, _ = image_np.shape

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    points = np.array(word.geometry)
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]

                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                    confidence = word.confidence
                    confidence_text = f"{confidence * 100:.0f}"

                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(annotated_image, confidence_text, (x_min, y_min - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image


def converter_pdf_para_imagens_fitz(pdf_bytes, output_prefix="imagem_pdf_"):
    """
    Converte PDF (em bytes) para imagens PNG usando PyMuPDF.
    Retorna lista com caminhos das imagens geradas.
    """
    caminhos = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        caminho = f"{output_prefix}{i + 1}.png"
        pix.save(caminho)
        caminhos.append(caminho)

    doc.close()
    return caminhos


def main():
    # Carregar modelo para ler pdf
    model_cadunico = load_ocr_model()

    # Carregar e exibir a imagem como cabeçalho
    header_image = Image.open("./img/header_sedu.jpeg")
    st.image(header_image, use_container_width=True)


    st.write("Demonstração dos modelos para o Projeto 11 SEDU")

    ####################################
    #############    RG   ##############

    st.subheader("Documento de Identidade")
    st.write("Envie um PDF de RG para processar e visualizar os resultados.")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="rg")

    if uploaded_file is not None:
        # Limpar imagens anteriores
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)

        # Converter PDF em imagens e exibir
        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())

        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        # Salvar PDF temporariamente para uso posterior
        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Botão para processar
        if st.button("Executar Modelos", key="btn_rg"):
            with st.spinner("Executando modelo..."):

                # Detecta RG
                imagens = detect_rg(temp_pdf_path)

                # Limpa recortes antigos
                pasta_recortes = "./recortes"
                for arquivo in os.listdir(pasta_recortes):
                    caminho = os.path.join(pasta_recortes, arquivo)
                    os.remove(caminho)

                # Divide RG
                dividir_rg(imagens)

                # Mostrar os recortes e as detecções
                file1 = "./recortes/img_1.png"
                file2 = "./recortes/img_2.png"

                data, meta_data_f, meta_data_v = etapa_final(file1, file2)

                # RG frente e verso
                file1 = "img_frente.png"
                file2 = "img_verso.png"


                if file1 and os.path.exists(file1):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(file1), caption="Frente", use_container_width=True)
                    with col2:
                        img_bbox = desenhar_bounding_boxes(file1, meta_data_f)
                        st.image(img_bbox, caption="Imagem com campos reconhecidos pela IA", use_container_width=True)

                if file2 and os.path.exists(file2):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(file2), caption="Verso", use_container_width=True)
                    with col2:
                        img_bbox = desenhar_bounding_boxes(file2, meta_data_v)
                        st.image(img_bbox, caption="Imagem com campos reconhecidos pela IA", use_container_width=True)

                # Mostrar os dados extraídos
                if isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=["Campos no documento", "Resultado identificado"])
                    st.table(df)
                else:
                    st.warning("Nenhum dado identificado no documento.")

            st.success("Processamento concluído!")

            # Limpar imagens anteriores
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)


    ####################################
    #############CADUNICO##############
    
    st.subheader("Cadúnico")
    st.write("Envie um PDF de CADUNICO para processar e visualizar os resultados.")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="cadunico")

    if uploaded_file is not None:
        # Limpar imagens anteriores
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)

        # Converter PDF em imagens e exibir
        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())

        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        # Salvar PDF temporariamente para uso posterior
        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Botão para processar
        if st.button("Executar Modelos", key="btn-cadunico"):
            with st.spinner("Executando modelo..."):

                
                data = process_document(model_cadunico, temp_pdf_path, show_image=False)

                # Mostrar os dados extraídos
                if isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=["Campos no documento", "Resultado identificado"])
                    df = df.explode("Resultado identificado", ignore_index=True)
                    st.table(df)

                else:
                    st.warning("Nenhum dado identificado no documento.")

            st.success("Processamento concluído!")
            # Limpar imagens anteriores
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)


    ####################################################
    #############Comprovante de residencia##############
    
    st.subheader("Comprovante de Residência")
    st.write("Envie um PDF de Comprovante de Residência para processar e visualizar os resultados.")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="residencia")

    if uploaded_file is not None:
        # Limpar imagens anteriores
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)
    
     # Converter PDF em imagens e exibir
        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())

        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        # Salvar PDF temporariamente para uso posterior
        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Botão para processar
        if st.button("Executar Modelos", key="btn-comp_res"):
            with st.spinner("Executando modelo..."):

                
                data = processar_pdf(temp_pdf_path, model_cadunico)

                # Mostrar os dados extraídos
                if isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=["Campos no documento", "Resultado identificado"])
                    st.table(df)
                else:
                    st.warning("Nenhum dado identificado no documento.")

            st.success("Processamento concluído!")
            # Limpar imagens anteriores
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)


if __name__ == "__main__":
    main()
