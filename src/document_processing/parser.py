"""Document parser using Docling and OCR for PDF and DOCX files."""

import hashlib
import io
from pathlib import Path

import easyocr
import fitz  # PyMuPDF
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger

from src.config import get_settings
from src.models import Document, DocumentStatus, DocumentType


class DocumentParser:
    """Parse documents (PDF, DOCX) to Markdown using Docling with OCR fallback."""

    def __init__(self, use_ocr: bool = True):
        self.settings = get_settings()
        self.use_ocr = use_ocr

        # Initialize EasyOCR reader for scanned PDFs
        if use_ocr:
            logger.info("Initializing EasyOCR reader for Vietnamese and English...")
            self._ocr_reader = easyocr.Reader(["vi", "en"], gpu=False)
        else:
            self._ocr_reader = None

        # Configure Docling converter
        self.converter = DocumentConverter()

        self._type_mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOC,
            ".pptx": DocumentType.PPTX,
            ".md": DocumentType.MARKDOWN,
            ".txt": DocumentType.TEXT,
        }

    def _generate_doc_id(self, filepath: Path) -> str:
        """Generate unique document ID from file path."""
        content = (
            f"{filepath.name}_{filepath.stat().st_size}_{filepath.stat().st_mtime}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_doc_type(self, filepath: Path) -> DocumentType:
        """Get document type from file extension."""
        suffix = filepath.suffix.lower()
        return self._type_mapping.get(suffix, DocumentType.TEXT)

    def _ocr_pdf(self, filepath: Path, dpi: int = 150) -> str:
        """
        OCR a scanned PDF using PyMuPDF + EasyOCR.

        Args:
            filepath: Path to PDF file
            dpi: Resolution for rendering pages

        Returns:
            Extracted text as markdown
        """
        if self._ocr_reader is None:
            raise ValueError("OCR reader not initialized")

        doc = fitz.open(str(filepath))
        all_text = []

        logger.info(f"OCR processing {len(doc)} pages...")

        for i, page in enumerate(doc):
            logger.debug(f"OCR page {i + 1}/{len(doc)}")

            # Render page to image
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")

            # OCR the image
            result = self._ocr_reader.readtext(img_data, detail=0, paragraph=True)

            # Add page header and content
            page_text = f"\n## Trang {i + 1}\n\n"
            page_text += "\n\n".join(result)
            all_text.append(page_text)

        doc.close()

        return "\n".join(all_text)

    def _is_scanned_pdf(self, filepath: Path) -> bool:
        """Check if PDF is scanned (image-based) without text layer."""
        doc = fitz.open(str(filepath))

        # Check first 3 pages
        total_text = 0
        pages_to_check = min(3, len(doc))

        for i in range(pages_to_check):
            text = doc[i].get_text()
            total_text += len(text.strip())

        doc.close()

        # If less than 100 characters in first 3 pages, likely scanned
        return total_text < 100

    def parse_file(
        self, filepath: str | Path, output_dir: str | Path | None = None
    ) -> Document:
        """
        Parse a single document file to Markdown.

        Args:
            filepath: Path to the input file
            output_dir: Directory to save processed markdown (optional)

        Returns:
            Document object with metadata
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        doc_type = self._get_doc_type(filepath)
        doc_id = self._generate_doc_id(filepath)

        logger.info(f"Parsing document: {filepath.name} (type: {doc_type.value})")

        document = Document(
            id=doc_id,
            filename=filepath.name,
            doc_type=doc_type,
            status=DocumentStatus.PROCESSING,
            source_path=str(filepath),
        )

        try:
            # For markdown and text files, just read content
            if doc_type in (DocumentType.MARKDOWN, DocumentType.TEXT):
                markdown_content = filepath.read_text(encoding="utf-8")
            elif doc_type == DocumentType.PDF:
                # Check if PDF is scanned (image-based)
                if self.use_ocr and self._is_scanned_pdf(filepath):
                    logger.info(f"Detected scanned PDF, using OCR...")
                    markdown_content = self._ocr_pdf(filepath)
                else:
                    # Use Docling for native PDFs
                    result = self.converter.convert(str(filepath))
                    markdown_content = result.document.export_to_markdown()

                    # Fallback to OCR if Docling returns mostly images
                    if (
                        self.use_ocr
                        and markdown_content.count("<!-- image -->") > 5
                        and len(markdown_content.replace("<!-- image -->", "").strip())
                        < 200
                    ):
                        logger.info(
                            f"Docling returned mostly images, falling back to OCR..."
                        )
                        markdown_content = self._ocr_pdf(filepath)
            else:
                # Use Docling for DOCX, PPTX
                result = self.converter.convert(str(filepath))
                markdown_content = result.document.export_to_markdown()

            # Save processed markdown if output_dir provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(markdown_content, encoding="utf-8")
                document.processed_path = str(output_path)
                logger.info(f"Saved processed markdown to: {output_path}")

            document.status = DocumentStatus.COMPLETED
            document.metadata["content"] = markdown_content
            document.metadata["content_length"] = len(markdown_content)

        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            document.status = DocumentStatus.FAILED
            document.metadata["error"] = str(e)

        return document

    def parse_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        extensions: list[str] | None = None,
    ) -> list[Document]:
        """
        Parse all documents in a directory.

        Args:
            input_dir: Input directory containing documents
            output_dir: Output directory for processed markdown files
            extensions: List of file extensions to process (default: all supported)

        Returns:
            List of Document objects
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # Default extensions
        if extensions is None:
            extensions = list(self._type_mapping.keys())

        documents = []
        files = []

        # Collect all matching files
        for ext in extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            files.extend(input_dir.glob(f"*{ext}"))
            files.extend(input_dir.glob(f"**/*{ext}"))  # Recursive

        # Remove duplicates
        files = list(set(files))

        logger.info(f"Found {len(files)} files to process in {input_dir}")

        for filepath in files:
            try:
                doc = self.parse_file(filepath, output_dir)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

        successful = sum(1 for d in documents if d.status == DocumentStatus.COMPLETED)
        logger.info(f"Successfully parsed {successful}/{len(documents)} documents")

        return documents

    def get_content(self, document: Document) -> str:
        """Get the content of a parsed document."""
        if document.status != DocumentStatus.COMPLETED:
            raise ValueError(
                f"Document {document.id} is not completed (status: {document.status})"
            )

        # Try to get from metadata first
        if "content" in document.metadata:
            return document.metadata["content"]

        # Otherwise read from processed path
        if document.processed_path:
            return Path(document.processed_path).read_text(encoding="utf-8")

        raise ValueError(f"No content available for document {document.id}")
