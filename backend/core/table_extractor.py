
import logging
import re
import pdfplumber

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts financial tables from PDF documents.

    Supports:
        1. Standard pdfplumber table extraction.
        2. Coordinate-based row and column reconstruction.
        3. Financial table classification using page headings.
        4. Markdown conversion for downstream RAG.
    """

    def extract_tables(
        self,
        pdf_path,
        pages=None,
        table_settings=None,
        coordinate_configs=None
    ):
        """
        Extract tables from selected PDF pages.

        Args:
            pdf_path: Path to the PDF.
            pages: Optional list of 1-based page numbers.
            table_settings: Optional pdfplumber settings.
            coordinate_configs: Optional page-specific configurations.

        Returns:
            List of dictionaries containing table information.
        """

        results = []

        coordinate_configs = coordinate_configs or {}

        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)

            page_numbers = (
                range(1, total_pages + 1)
                if pages is None
                else pages
            )

            for page_num in page_numbers:

                if page_num < 1 or page_num > total_pages:

                    logger.warning(
                        "Skipping invalid page: %s",
                        page_num
                    )
                    continue

                page = pdf.pages[page_num - 1]

                page_text = page.extract_text() or ""

                # Check whether this page has a coordinate profile.
                if page_num in coordinate_configs:

                    config = coordinate_configs[page_num]

                    table = self._extract_coordinate_table(
                        page=page,
                        config=config
                    )

                    extracted_tables = (
                        [table] if table else []
                    )

                else:

                    # Standard pdfplumber extraction.
                    if table_settings is None:

                        extracted_tables = page.extract_tables()

                    else:

                        extracted_tables = page.extract_tables(
                            table_settings=table_settings
                        )

                for idx, table in enumerate(extracted_tables):

                    cleaned = self.clean_table(table)

                    if cleaned is None:
                        continue

                    # Standard table extraction may have split headers.
                    if page_num not in coordinate_configs:

                        cleaned = self._normalize_year_header(
                            cleaned
                        )

                    table_type = self.classify_table_type(
                        cleaned,
                        page_text=page_text
                    )

                    results.append({

                        "page_num": page_num,

                        "table_index": idx,

                        "headers": cleaned[0],

                        "rows": cleaned[1:],

                        "table_type": table_type,

                        "text": self.table_to_text(
                            cleaned,
                            page_num
                        )
                    })

        logger.info(
            "Found %s tables",
            len(results)
        )

        return results

    def _extract_coordinate_table(self, page, config):
        """
        Reconstructs a table using individual word coordinates.

        This method is useful for financial PDFs where:
        - Vertical borders are absent.
        - Automatic table detection splits columns.
        - Financial values are aligned by position.

        Configuration:
            column_boundaries: Three X-coordinate boundaries.
            top: Top Y-coordinate of the table area.
            bottom: Bottom Y-coordinate of the table area.
            y_tolerance: Maximum vertical grouping difference.
        """

        boundaries = config.get("column_boundaries")

        top = config.get("top")
        bottom = config.get("bottom")

        y_tolerance = config.get(
            "y_tolerance",
            2
        )

        if not boundaries or len(boundaries) != 3:

            raise ValueError(
                "Coordinate extraction requires exactly "
                "three column boundaries."
            )

        if top is None or bottom is None or top >= bottom:

            raise ValueError(
                "Invalid coordinate extraction area."
            )

        if boundaries != sorted(boundaries):

            raise ValueError(
                "Column boundaries must be in ascending order."
            )

        # Extract individual words and their PDF coordinates.
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3
        )

        selected_words = [
            word for word in words
            if top <= word["top"] <= bottom
        ]

        if not selected_words:
            return None

        # Sort words vertically and then horizontally.
        selected_words.sort(
            key=lambda word: (
                (word["top"] + word["bottom"]) / 2,
                word["x0"]
            )
        )

        rows = []

        # Group words into rows using their Y-coordinates.
        for word in selected_words:

            center_y = (
                word["top"] + word["bottom"]
            ) / 2

            matching_row = None

            for row in reversed(rows):

                if abs(row["center_y"] - center_y) <= y_tolerance:

                    matching_row = row
                    break

                if row["center_y"] < center_y - y_tolerance:
                    break

            if matching_row is None:

                matching_row = {

                    "center_y": center_y,

                    "columns": [
                        [],
                        [],
                        [],
                        []
                    ]
                }

                rows.append(matching_row)

            # Assign words to their respective columns.
            x0 = word["x0"]

            column_index = sum(
                x0 >= boundary
                for boundary in boundaries
            )

            matching_row["columns"][column_index].append(word)

        # Convert grouped words into four-column rows.
        reconstructed_rows = []

        for row in rows:

            values = []

            for column_words in row["columns"]:

                column_words.sort(
                    key=lambda word: word["x0"]
                )

                text = " ".join(
                    word["text"]
                    for word in column_words
                )

                text = " ".join(text.split())

                values.append(text)

            if any(values):

                reconstructed_rows.append(values)

        # Identify the column headings.
        header_index = next(
            (
                index
                for index, row in enumerate(reconstructed_rows)
                if "particulars" in row[0].lower()
            ),
            None
        )

        if header_index is None:

            logger.warning(
                "Could not find Particulars header."
            )

            return None

        header_row = reconstructed_rows[header_index]

        # Find the row containing the financial years.
        year_index = None

        for index in range(
            header_index + 1,
            len(reconstructed_rows)
        ):

            row = reconstructed_rows[index]

            year1 = re.search(
                r"\b(19|20)\d{2}\b",
                row[2]
            )

            year2 = re.search(
                r"\b(19|20)\d{2}\b",
                row[3]
            )

            if year1 and year2:

                year_index = index
                break

        if year_index is None:

            logger.warning(
                "Could not identify financial year columns."
            )

            return None

        year_row = reconstructed_rows[year_index]

        year1 = re.search(
            r"\b(?:19|20)\d{2}\b",
            year_row[2]
        ).group()

        year2 = re.search(
            r"\b(?:19|20)\d{2}\b",
            year_row[3]
        ).group()

        # Combine period heading fragments.
        period_text = " ".join(
            header_row[2:4]
        )

        period_text = re.sub(
            r"\s+",
            " ",
            period_text
        ).strip(" ,")

        headers = [

            header_row[0] or "Particulars",

            header_row[1] or "Note",

            f"{year1} ({period_text})",

            f"{year2} ({period_text})"
        ]

        # Everything after the year heading is table data.
        data_rows = reconstructed_rows[year_index + 1:]

        return [
            headers,
            *data_rows
        ]

    def clean_table(self, table):
        """
        Cleans table rows and cells.

        Removes empty rows and normalizes whitespace.
        """

        cleaned = []

        for row in table:

            new_row = [

                " ".join(
                    str(cell or "")
                    .replace("\n", " ")
                    .split()
                )

                for cell in row
            ]

            if any(new_row):

                cleaned.append(new_row)

        if len(cleaned) < 2 or len(cleaned[0]) < 2:

            return None

        return cleaned

    def _normalize_year_header(self, table):
        """
        Combines split financial year headings.

        Used by the standard pdfplumber extraction path.
        """

        if len(table) < 2:
            return table

        first_row = list(table[0])

        second_row = list(table[1])

        if len(first_row) != 4 or len(second_row) != 4:
            return table

        period_text = " ".join(
            first_row[2:]
        ).strip()

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

            return [
                normalized_header,
                *table[2:]
            ]

        return table

    def classify_table_type(self, table, page_text=None):
        """
        Classifies financial tables using page headings first.

        Uses table content as a fallback.
        """

        heading = "\n".join(
            (page_text or "").splitlines()[:8]
        ).lower()

        # Page heading is more reliable than individual keywords.
        if "statement of profit and loss" in heading:

            return "income_statement"

        if "balance sheet" in heading:

            return "balance_sheet"

        if "cash flow" in heading:

            return "cash_flow"

        if "statement of changes in equity" in heading:

            return "equity"

        # Fallback to table content.
        text = " ".join(
            " ".join(row)
            for row in table
        ).lower()

        if (
            "profit" in text
            and ("revenue" in text or "expenses" in text)
        ):

            return "income_statement"

        if (
            "total assets" in text
            or "total liabilities" in text
            or "total equity and liabilities" in text
        ):

            return "balance_sheet"

        if (
            "cash flow" in text
            or "operating activities" in text
        ):

            return "cash_flow"

        if (
            "statement of changes in equity" in text
        ):

            return "equity"

        return "other"

    def table_to_text(self, table, page_num):
        """
        Converts a financial table into Markdown text.

        This text will later be embedded and stored in Qdrant.
        """

        lines = [
            f"Financial table from page {page_num}:"
        ]

        for index, row in enumerate(table):

            lines.append(
                "| " + " | ".join(row) + " |"
            )

            if index == 0:

                lines.append(
                    "|" + "---|" * len(row)
                )

        return "\n".join(lines)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    extractor = TableExtractor()

    pdf_path = input(
        "Enter PDF path: "
    ).strip()

    page = int(
        input(
            "Enter ONE page number to test (e.g. 272): "
        ).strip()
    )

    tables = extractor.extract_tables(
        pdf_path,
        pages=[page]
    )

    print(
        f"\nTables found on page {page}: {len(tables)}"
    )

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
