from pathlib import Path
from pdf2image import convert_from_path

GOOGLE_DRIVE_PATH = Path("/Users/apple/Google Drive/マイドライブ/pj-takase")


def convert_pdf_to_images(save_path: Path) -> None:
    for pdf in GOOGLE_DRIVE_PATH.glob("*"):
        convert_from_path(
            pdf, output_folder=save_path, fmt="jpeg", output_file=pdf.stem, dpi=200
        )
