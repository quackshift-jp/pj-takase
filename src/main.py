from module import pdf_to_image
from pathlib import Path

if __name__ == "__main__":
    pdf_to_image.convert_pdf_to_images(Path("./images"))
