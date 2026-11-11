import logging
import pdfplumber

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Finds tables inside a PDF and converts them into clean text
    that can be embedded and searched, just like normal text.
    """

    def extract_tables(self, pdf_path, pages=None):
        """
        Extracts tables from a PDF.
        pages = list of page numbers to scan, e.g. [272, 273].
        If pages is None, every page is scanned (slow for big PDFs).
        """
        results = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            if pages is None:
                page_numbers = range(1, total_pages + 1)
            else:
                page_numbers = pages

            for page_num in page_numbers:
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()

                for idx, table in enumerate(tables):
                    cleaned = self.clean_table(table)

                    # Skip tables that are empty or too small
                    if cleaned is None:
                        continue

                    results.append({
                        "page_num": page_num,
                        "table_index": idx,
                        "headers": cleaned[0],
                        "rows": cleaned[1:],
                        "table_type": self.classify_table_type(cleaned),
                        "text": self.table_to_text(cleaned, page_num)
                    })

        logger.info(f"Found {len(results)} tables")
        return results

    def clean_table(self, table):
        """
        Removes empty rows and replaces None cells with "".
        Returns None if the table is too small to be useful.
        """
        cleaned = []

        for row in table:
            # Replace None with empty string and strip spaces
            new_row = [
                (cell or "").replace("\n", " ").strip()
                for cell in row
            ]

            # Keep the row only if it has at least one filled cell
            if any(new_row):
                cleaned.append(new_row)

        # A useful table needs at least 2 rows and 2 columns
        if len(cleaned) < 2 or len(cleaned[0]) < 2:
            return None

        return cleaned

    def classify_table_type(self, table):
        """
        Guesses what kind of financial table this is
        by looking for keywords in its text.
        """
        text = " ".join(" ".join(row) for row in table).lower()

        if "profit" in text and ("revenue" in text or "expenses" in text):
            return "income_statement"
        if "total assets" in text or "total liabilities" in text:
            return "balance_sheet"
        if "cash flow" in text or "operating activities" in text:
            return "cash_flow"
        if "equity" in text and "shares" in text:
            return "equity"
        return "other"

    def table_to_text(self, table, page_num):
        """
        Converts a table into markdown-style text.
        This text is what we will embed and store in Qdrant.
        """
        lines = [f"Financial table from page {page_num}:"]

        for i, row in enumerate(table):
            lines.append("| " + " | ".join(row) + " |")

            # Add a separator line after the header row
            if i == 0:
                lines.append("|" + "---|" * len(row))

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = TableExtractor()

    pdf_path = input("Enter PDF path: ").strip()
    page = int(input("Enter ONE page number to test (e.g. 272): ").strip())

    tables = extractor.extract_tables(pdf_path, pages=[page])

    print(f"\nTables found on page {page}: {len(tables)}")

    for t in tables:
        print("\n" + "=" * 50)
        print(f"Table {t['table_index']} | Type: {t['table_type']}")
        print(f"Columns: {len(t['headers'])} | Rows: {len(t['rows'])}")
        print("=" * 50)
        print(t["text"][:800])