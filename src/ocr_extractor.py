"""OCR module — extracts text from images for downstream NLP classification."""
import io
import base64
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def extract_from_path(image_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    if not _OCR_AVAILABLE:
        return ''
    try:
        img = Image.open(image_path).convert('RGB')
        return pytesseract.image_to_string(img, config='--psm 6').strip()
    except Exception:
        return ''


def extract_from_bytes(image_bytes: bytes) -> str:
    """Extract text from raw image bytes (e.g. from an HTTP upload)."""
    if not _OCR_AVAILABLE:
        return ''
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return pytesseract.image_to_string(img, config='--psm 6').strip()
    except Exception:
        return ''


def extract_from_base64(b64_string: str) -> str:
    """Extract text from a base64-encoded image string."""
    try:
        img_bytes = base64.b64decode(b64_string)
        return extract_from_bytes(img_bytes)
    except Exception:
        return ''


def ocr_available() -> bool:
    return _OCR_AVAILABLE
