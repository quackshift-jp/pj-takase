from module import pdf_to_image
from pathlib import Path
from ocr import google_vision

if __name__ == "__main__":
    pdf_to_image.convert_pdf_to_images(Path("./images"))
    image = google_vision.read_image()
    response = google_vision.extract_text(image)
    print(response["fullTextAnnotation"]["text"].replace("\n", ""))
