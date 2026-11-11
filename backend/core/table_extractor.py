
import logging
import re
import pdfplumber

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Finds tables inside a PDF and converts them into clean text
    that can be embedded and searched, just like normal text.
    """

    def extract_tables(self, pdf_path, pages=None, table_settings=None):
        """
        Extract tables from a PDF.

        Args:
            pdf_path: Path to the PDF.
            pages: List of 1-based page numbers to scan.
            table_settings: Optional pdfplumber table detection settings.

        Returns:
            List of dictionaries containing extracted tables.
        """

        results = []

        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)

            if pages is None:
                page_numbers = range(1, total_pages + 1)
            else:
                page_numbers = pages

            for page_num in page_numbers:

                if page_num < 1 or page_num > total_pages:
                    logger.warning(
                        "Skipping invalid page number: %s",
                        page_num
                    )
                    continue

                page = pdf.pages[page_num - 1]

                # Use custom settings when supplied.
                # Otherwise, use pdfplumber's default settings.
                if table_settings is None:
                    tables = page.extract_tables()
                else:
                    tables = page.extract_tables(
                        table_settings=table_settings
                    )

                for idx, table in enumerate(tables):

                    cleaned = self.clean_table(table)

                    if cleaned is None:
                        continue

                    # Normalize financial year headers.
                    cleaned = self._normalize_year_header(cleaned)

                    results.append({
                        "page_num": page_num,
                        "table_index": idx,
                        "headers": cleaned[0],
                        "rows": cleaned[1:],
                        "table_type": self.classify_table_type(cleaned),
                        "text": self.table_to_text(cleaned, page_num)
                    })

        logger.info("Found %s tables", len(results))

        return results

    def clean_table(self, table):
        """
        Removes empty rows and replaces None cells with empty strings.
        Normalizes whitespace and removes completely empty rows.
        """

        cleaned = []

        for row in table:

            new_row = [
                " ".join((cell or "").replace("\n", " ").split())
                for cell in row
            ]

            if any(new_row):
                cleaned.append(new_row)

        if len(cleaned) < 2 or len(cleaned[0]) < 2:
            return None

        return cleaned

    def _normalize_year_header(self, table):
        """
        Combines a split financial year heading.

        Example:
            ['Particulars', 'Note', 'Year ended M', 'arch 31,']
            ['', '', '2024', '2023']

        Becomes:
            ['Particulars', 'Note',
             '2024 (Year ended March 31)',
             '2023 (Year ended March 31)']
        """

        if len(table) < 2:
            return table

        first_row = list(table[0])
        second_row = list(table[1])

        if len(first_row) != 4 or len(second_row) != 4:
            return table

        period_text = " ".join(first_row[2:]).strip()

        # Rejoin words split across adjacent cells.
        period_text = re.sub(
            r"\b([A-Z])\s+([a-z]+)\b",
            r"\1\2",
            period_text
        )

        period_text = period_text.strip(" ,")

        years = [
            cell.strip()
            for cell in second_row[2:]
        ]

        valid_years = (
            len(years) == 2
            and all(
                re.fullmatch(r"\d{4}", year)
                for year in years
            )
        )

        if (
            "year ended" in period_text.lower()
            and "march" in period_text.lower()
            and valid_years
        ):

            normalized_header = (
                first_row[:2]
                + [
                    f"{year} ({period_text})"
                    for year in years
                ]
            )

            return [normalized_header] + table[2:]

        return table

    def classify_table_type(self, table):
        """
        Identifies the financial table category using keywords.
        """

        text = " ".join(
            " ".join(row)
            for row in table
        ).lower()

        if "profit" in text and (
            "revenue" in text or "expenses" in text
        ):
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
        Converts an extracted table into Markdown text.

        This output will later be embedded into Qdrant.
        """

        lines = [
            f"Financial table from page {page_num}:"
        ]

        for i, row in enumerate(table):

            lines.append(
                "| " + " | ".join(row) + " |"
            )

            if i == 0:
                lines.append(
                    "|" + "---|" * len(row)
                )

        return "\n".join(lines)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    extractor = TableExtractor()

    pdf_path = input("Enter PDF path: ").strip()

    page = int(
        input("Enter ONE page number to test (e.g. 272): ").strip()
    )

    tables = extractor.extract_tables(
        pdf_path,
        pages=[page]
    )

    print(f"\nTables found on page {page}: {len(tables)}")

    for table in tables:

        print("\n" + "=" * 50)

        print(
            f"Table {table['table_index']} | "
            f"Type: {table['table_type']}"
        )

        print(
            f"Columns: {len(table['headers'])} | "
            f"Rows: {len(table['rows'])}"
        )

        print("=" * 50)

        print(table["text"])
