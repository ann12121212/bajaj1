import os
import requests
from io import BytesIO
from langchain_community.document_loaders import UnstructuredEmailLoader, Docx2txtLoader, PDFPlumberLoader
import urllib.parse
from pathlib import Path
import tempfile

def loading(file_path: str):
    # Handle remote files (currently only PDF)
    if file_path.startswith(('http://', 'https://')):
        response = requests.get(file_path)
        response.raise_for_status()

        if file_path.endswith(".pdf"):
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            try:
                return PDFPlumberLoader(tmp_path).load()
            finally:
                os.remove(tmp_path)  # Clean up the temp file
        else:
            raise ValueError("URL loading only supports PDFs")

    # Decode URL-encoded local paths
    file_path = urllib.parse.unquote(file_path)
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle local file loading
    if file_path.endswith(".pdf"):
        loader = PDFPlumberLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".eml"):
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only .pdf, .docx, and .eml are supported.")
    return loader.load()
