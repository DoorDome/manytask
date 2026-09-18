from typing import Any


def update_cells_request(sheet_id: int, row: int, column: int, values: list[Any]) -> dict[str, Any]:
    return {"updateCells": {
        "start": {"sheetId": sheet_id, "rowIndex": row - 1, "columnIndex": column - 1},
        "rows": [{"values": [{"userEnteredValue": (
            {"numberValue": value} if isinstance(value, (int, float)) else {"stringValue": str(value)}
        )} for value in values]}],
        "fields": "userEnteredValue",
    }}
