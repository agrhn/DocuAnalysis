import PyPDF2
from langchain_core.documents import Document
import os
from typing import List


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and convert it to a list of Document objects.
    Each page becomes a separate Document with metadata.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        List[Document]: List of Document objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")

    documents: List[Document] = []

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
                text = page.extract_text()

                # Create a Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "filename": file_name,
                        "page": i + 1,
                        "total_pages": num_pages,
                        "title": file_name_no_ext
                    }
                )
                documents.append(doc)

    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")

    return documents
