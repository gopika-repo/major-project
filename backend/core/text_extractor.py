import fitz  # this is PyMuPDF
import os
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    """Reads PDF files and extracts text page by page."""

    def extract_text_from_pdf(self, pdf_path):
        """
        Opens a PDF and extracts text from every page.
        Returns a list of dictionaries — one per page.
        """
        pages = []

        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                # Load one page at a time
                page = doc.load_page(page_num)

                # Extract raw text from the page
                text = page.get_text("text")

                # Clean the text
                text = self.clean_text(text)

                # Count words
                word_count = len(text.split())

                # Skip blank pages (less than 10 words)
                if word_count < 10:
                    continue

                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "word_count": word_count
                })

            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
            return pages

        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            raise

    def extract_with_metadata(self, pdf_path):
        """
        Same as extract_text_from_pdf but also detects
        which section each page belongs to.
        """
        pages = []
        current_section = "UNKNOWN"

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text = self.clean_text(text)
                word_count = len(text.split())

                if word_count < 10:
                    continue

                # Detect section heading
                # A heading = line that is ALL CAPS and less than 8 words
                for line in text.split("\n"):
                    line = line.strip()
                    if line.isupper() and len(line.split()) < 8 and len(line) > 3:
                        current_section = line
                        break

                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "section": current_section,
                    "word_count": word_count
                })

            doc.close()
            logger.info(f"Extracted {len(pages)} pages with metadata")
            return pages

        except Exception as e:
            logger.error(f"Failed to extract with metadata: {e}")
            raise

    def clean_text(self, text):
        """
        Cleans raw text extracted from PDF.
        Removes extra spaces, blank lines, junk characters.
        """
        import re

        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove lone page numbers like "1" or "Page 1"
        text = re.sub(r'\bPage\s+\d+\b', '', text)

        # Remove null characters
        text = text.replace('\x00', '')

        # Strip extra whitespace
        text = text.strip()

        return text

    def get_pdf_info(self, pdf_path):
        """
        Returns basic info about a PDF without
        extracting full text.
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            file_size = os.path.getsize(pdf_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            filename = os.path.basename(pdf_path)

            return {
                "num_pages": num_pages,
                "file_size_mb": file_size_mb,
                "filename": filename
            }

        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = TextExtractor()
    pdf_path = input("Enter path to your PDF: ").strip()

    print("\n--- PDF Info ---")
    info = extractor.get_pdf_info(pdf_path)
    print(f"File     : {info['filename']}")
    print(f"Pages    : {info['num_pages']}")
    print(f"Size     : {info['file_size_mb']} MB")

    print("\n--- Extracting text ---")
    pages = extractor.extract_with_metadata(pdf_path)

    for page in pages[:3]:
        print(f"\nPage {page['page_num']} | Section: {page['section']} | Words: {page['word_count']}")
        print(f"Preview: {page['text'][:100]}...")

    total_words = sum(p['word_count'] for p in pages)
    print(f"\nTotal: {len(pages)} pages | {total_words} words ✅")