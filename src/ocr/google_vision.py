from google.cloud import vision
import io
from google.oauth2 import service_account
from google.cloud.vision_v1 import AnnotateImageResponse
import json

CREDENTIAL = service_account.Credentials.from_service_account_file("key.json")
CLIENT = vision.ImageAnnotatorClient(credentials=CREDENTIAL)
IMAGE_PATH = "/Users/apple/src/quackshift/pj-takase/src/ocr/請求書0001-1.jpg"


def read_image(image_path: str = IMAGE_PATH) -> vision.Image:
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    return vision.Image(content=content)


def extract_text(image: vision.Image) -> dict[str, any]:
    response = CLIENT.document_text_detection(
        image=image, image_context={"language_hints": ["ja"]}
    )
    return json.loads(AnnotateImageResponse.to_json(response))
