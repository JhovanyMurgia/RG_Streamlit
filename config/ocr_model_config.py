# config/ocr_model_config.py

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def load_ocr_model(classifier=False):
    if classifier:
        return ocr_predictor(pretrained=True, assume_straight_pages=True)
    return ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True, assume_straight_pages=False)

def run_ocr(model, image_path, show_image=False):
    document = DocumentFile.from_images(image_path)
    result = model(document)
    if show_image:
        result.show()
    return result