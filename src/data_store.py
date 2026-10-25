"""Read the club's stock picks from a CSV file.

CSV is a spreadsheet saved as plain text. You can open data/picks.csv
in Excel or Numbers, edit students/tickers, save, then refresh the app.
"""

import csv
from pathlib import Path

from src.config import PICKS_CSV


def load_picks(path: Path | None = None) -> list[dict[str, str]]:
    """Return rows like {"student": "Annie Shen", "company": "Nvidia", "ticker": "NVDA"}."""
    csv_path = path or PICKS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Create data/picks.csv with columns student,ticker."
        )

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("picks.csv has no header row. First line should be: student,company,ticker")

        raw_headers = [name.strip().lower() for name in reader.fieldnames]
        headers = set(raw_headers)
        if "ticker" not in headers:
            raise ValueError("picks.csv must have a ticker column.")
        if "student" not in headers and "team" not in headers:
            raise ValueError("picks.csv must have a student column (or team).")

        picks = []
        for row in reader:
            lowered = {key.strip().lower(): (value or "").strip() for key, value in row.items()}
            student = lowered.get("student") or lowered.get("team")
            company = lowered.get("company") or ""
            ticker = (lowered.get("ticker") or "").upper()
            if student and ticker:
                picks.append({"student": student, "company": company, "ticker": ticker})
        return picks


def unique_tickers(picks: list[dict[str, str]]) -> list[str]:
    """Sorted list of tickers with duplicates removed."""
    return sorted({row["ticker"] for row in picks})
