import PyPDF2
import os
from typing import Tuple


def load_pdf(file_path: str) -> Tuple[str, dict]:
    """
    Load a PDF file, extract the text, and combine it into a single string.
    Returns the combined text and metadata about the document.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        Tuple[str, dict]: A tuple containing:
            - A single string with all the text from the PDF
            - Metadata about the document
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")

    combined_text = ""
    metadata = {}

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            # Extract filename without extension for metadata
            file_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(file_name)[0]

            # Process each page
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()

                # Combine all page text into a single string
                combined_text += page_text + "\n"

            # Populate metadata
            metadata = {
                "source": file_path,
                "filename": file_name,
                "total_pages": num_pages,
                "title": file_name_no_ext
            }

    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")

    return combined_text.strip(), metadata
