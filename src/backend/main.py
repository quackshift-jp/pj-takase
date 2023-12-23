from module import pdf_to_image
from pathlib import Path
from ocr import google_vision, azure_computer_vision

if __name__ == "__main__":
    pdf_to_image.convert_pdf_to_images(Path("./images"), search_keyword="重要事項説明書")
    # image = google_vision.read_image("images/重要事項説明書0001-1.jpg")
    # response = google_vision.extract_text(image)
    # print(response["fullTextAnnotation"]["text"].replace("\n", ""))

    operation_id = azure_computer_vision.get_operation_id(
        "images/234119　重要事項説明書0001-1.jpg"
    )
    text = azure_computer_vision.extract_text(operation_id)
    print(text)
