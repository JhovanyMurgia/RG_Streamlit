import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["USE_TF"] = "FALSE"

import torch
torch.classes.__path__ = []

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import fitz
from ultralytics import YOLO

from pipeline.full_pipeline import detect_rg, split_rg, etapa_final
from ocr.modelo_cadunico import load_ocr_model_cadunico, process_document
from ocr.modelo_comp_resid import processar_pdf
from config.ocr_model_config import load_ocr_model

@st.cache_resource
def carregar_modelo_yolo():
    return YOLO("./weights/best.pt")

@st.cache_resource
def carregar_modelo_ocr_cadunico():
    return load_ocr_model_cadunico()

@st.cache_resource
def carregar_modelo_classifier():
    return load_ocr_model(classifier=True)

@st.cache_resource
def carregar_modelo_sem_classificador():
    return load_ocr_model(classifier=False)

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
    model_yolo = carregar_modelo_yolo()
    model_cadunico = carregar_modelo_ocr_cadunico()
    model_classifier = carregar_modelo_classifier()
    model = carregar_modelo_sem_classificador()

    header_image = Image.open("./img/header_sedu.jpeg")
    st.image(header_image, use_container_width=True)

    st.write("Demonstração dos modelos para o Projeto 11 SEDU")

    st.subheader("Documento de Identidade")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="rg")

    if uploaded_file is not None:
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)

        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())
        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Executar Modelos", key="btn_rg"):
            with st.spinner("Executando modelo..."):
                imagens = detect_rg(temp_pdf_path, model_yolo)
                for arquivo in os.listdir("./recortes"):
                    os.remove(os.path.join("./recortes", arquivo))
                split_rg(imagens)

                file1 = "./recortes/img_1.png"
                file2 = "./recortes/img_2.png"
                data, meta_data_f, meta_data_v = etapa_final(file1, file2, model_classifier, model)

                file1_out = "img_frente.png"
                file2_out = "img_verso.png"

                if os.path.exists(file1_out):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(file1_out), caption="Frente", use_container_width=True)
                    with col2:
                        img_bbox = desenhar_bounding_boxes(file1_out, meta_data_f)
                        st.image(img_bbox, caption="Imagem com campos reconhecidos", use_container_width=True)

                if os.path.exists(file2_out):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(file2_out), caption="Verso", use_container_width=True)
                    with col2:
                        img_bbox = desenhar_bounding_boxes(file2_out, meta_data_v)
                        st.image(img_bbox, caption="Imagem com campos reconhecidos", use_container_width=True)

                if isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=["Campos no documento", "Resultado identificado"])
                    st.table(df)
                else:
                    st.warning("Nenhum dado identificado no documento.")

            st.success("Processamento concluído!")
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)

    st.subheader("Cadúnico")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="cadunico")

    if uploaded_file is not None:
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)

        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())
        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Executar Modelos", key="btn-cadunico"):
            with st.spinner("Executando modelo..."):
                data = process_document(model_cadunico, temp_pdf_path, show_image=False)
                if isinstance(data, dict):
                    if any(isinstance(v, list) for v in data.values()):
                        max_len = max(len(v) if isinstance(v, list) else 1 for v in data.values())
                        normalized_data = {k: v if isinstance(v, list) else [v] * max_len for k, v in data.items()}
                        df = pd.DataFrame(normalized_data)
                    else:
                        df = pd.DataFrame([data])
                    st.table(df)
                else:
                    st.warning("Nenhum dado identificado no documento.")
            st.success("Processamento concluído!")
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)

    st.subheader("Comprovante de Residência")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"], key="residencia")

    if uploaded_file is not None:
        for f in os.listdir('.'):
            if f.startswith("imagem_pdf_") and f.endswith(".png"):
                os.remove(f)

        st.subheader("Visualização do PDF enviado:")
        imagens_convertidas = converter_pdf_para_imagens_fitz(uploaded_file.getvalue())
        for caminho_img in imagens_convertidas:
            st.image(Image.open(caminho_img), caption=caminho_img, use_container_width=True)

        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Executar Modelos", key="btn-comp_res"):
            with st.spinner("Executando modelo..."):
                data = processar_pdf(temp_pdf_path, model_cadunico)
                if isinstance(data, dict):
                    filtered = {k: v for k, v in data.items() if v not in [None, '', [], {}, 'null']}
                    if filtered:
                        df = pd.DataFrame(list(filtered.items()), columns=["Campos no documento", "Resultado identificado"])
                        st.table(df)
                    else:
                        st.warning("Nenhum dado identificado no documento.")
                else:
                    st.warning("Nenhum dado identificado no documento.")
            st.success("Processamento concluído!")
            for f in os.listdir('.'):
                if f.endswith(".png"):
                    os.remove(f)

if __name__ == "__main__":
    main()
