import sys
from io import BytesIO
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from customer_analytics.ingestion import load_uploaded_file, IngestionError


def test_csv_ingestion():
    raw = b"Customer_ID,Order_Value\nC1,12\nC2,15\n"
    result = load_uploaded_file("sample.csv", raw)
    assert len(result) == 1
    assert result[0].dataframe.shape == (2, 2)
    assert result[0].source_format == "csv"


def test_xlsx_ingestion_multiple_sheets():
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Customer_ID": ["C1"]}).to_excel(writer, sheet_name="Customers", index=False)
        pd.DataFrame({"Order_ID": ["O1"]}).to_excel(writer, sheet_name="Orders", index=False)
    loaded = load_uploaded_file("sample.xlsx", buffer.getvalue())
    assert [x.table_name for x in loaded] == ["Customers", "Orders"]


def test_reject_unsupported_format_and_size_limit():
    with pytest.raises(IngestionError, match="Unsupported"):
        load_uploaded_file("sample.json", b"{}")
    with pytest.raises(IngestionError, match="limit"):
        load_uploaded_file("sample.csv", b"abc", max_file_mb=0)
