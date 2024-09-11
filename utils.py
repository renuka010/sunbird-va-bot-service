import base64
import binascii
import uuid
from urllib.parse import urlparse


def is_base64(base64_string):
    try:
        base64.b64decode(base64_string)
        return True
    except (binascii.Error, UnicodeDecodeError):
        # If an error occurs during decoding, the string is not Base64
        return False


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def generate_temp_filename(ext, prefix = "temp"):
    return f"{prefix}_{uuid.uuid4()}.{ext}"