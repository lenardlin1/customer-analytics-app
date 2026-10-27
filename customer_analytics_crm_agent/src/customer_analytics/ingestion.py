"""Controlled ingestion for CSV and Excel uploads."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import zipfile

import pandas as pd

DEFAULT_MAX_FILE_MB = 50
MAX_EXCEL_SHEETS = 100
MAX_EXCEL_UNCOMPRESSED_MB = 250


class IngestionError(ValueError):
    """Raised when an uploaded file cannot be read safely or reliably."""


@dataclass
class LoadedTable:
    source_name: str
    table_name: str
    dataframe: pd.DataFrame
    source_format: str


def load_uploaded_file(
    filename: str,
    file_bytes: bytes,
    *,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> list[LoadedTable]:
    """Load a CSV or XLSX upload without writing it to disk.

    All worksheets in an XLSX workbook are returned. The original bytes are not
    modified. Header-row detection remains explicit: pandas' default first-row
    header is used, so unusual worksheets should be confirmed by the user.
    """
    if not filename or not file_bytes:
        raise IngestionError("The uploaded file is empty or has no filename.")
    max_bytes = max_file_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise IngestionError(
            f"File is {len(file_bytes) / (1024 * 1024):.1f} MB; "
            f"the configured limit is {max_file_mb} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            try:
                frame = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="utf-8-sig")
            except UnicodeDecodeError:
                frame = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="cp1252")
        except Exception as exc:  # parsing libraries can raise several exception types
            raise IngestionError(f"Could not parse CSV: {exc}") from exc
        frame.columns = [str(column) for column in frame.columns]
        return [LoadedTable(filename, Path(filename).stem, frame, "csv")]

    if suffix == ".xlsx":
        if not zipfile.is_zipfile(BytesIO(file_bytes)):
            raise IngestionError("The .xlsx file does not have a valid ZIP-based Excel structure.")
        try:
            with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
                expanded_bytes = sum(info.file_size for info in archive.infolist())
            expanded_limit = MAX_EXCEL_UNCOMPRESSED_MB * 1024 * 1024
            if expanded_bytes > expanded_limit:
                raise IngestionError(
                    f"Workbook expands to {expanded_bytes / (1024 * 1024):.1f} MB; "
                    f"the configured uncompressed-content limit is {MAX_EXCEL_UNCOMPRESSED_MB} MB."
                )
            workbook = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
            if len(workbook.sheet_names) > MAX_EXCEL_SHEETS:
                raise IngestionError(
                    f"Workbook has {len(workbook.sheet_names)} sheets; limit is {MAX_EXCEL_SHEETS}."
                )
            tables: list[LoadedTable] = []
            for sheet in workbook.sheet_names:
                try:
                    frame = pd.read_excel(workbook, sheet_name=sheet)
                except Exception as exc:
                    # Keep the rest of the workbook available instead of silently hiding a bad sheet.
                    raise IngestionError(f"Could not read worksheet {sheet!r}: {exc}") from exc
                frame.columns = [str(column) for column in frame.columns]
                tables.append(LoadedTable(filename, str(sheet), frame, "xlsx"))
            return tables
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Could not read Excel workbook: {exc}") from exc

    raise IngestionError("Unsupported file type. Upload a .csv or .xlsx file.")
