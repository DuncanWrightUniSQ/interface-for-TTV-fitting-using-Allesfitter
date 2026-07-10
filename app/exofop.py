"""ExoFOP-TESS lookup helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd


TOI_TABLE_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"
TOI_CACHE_DIR = Path("data") / "exofop"
TOI_CACHE_PATH = TOI_CACHE_DIR / "toi_list.csv"
TOI_HISTORY_PATH = TOI_CACHE_DIR / "toi_list_history.csv"


PREFERRED_TOI_COLUMNS = [
    "TIC ID",
    "TOI",
    "Planet Name",
    "TESS Disposition",
    "TFOPWG Disposition",
    "Epoch (BJD)",
    "Epoch (BJD) err",
    "Period (days)",
    "Period (days) err",
    "Duration (hours)",
    "Duration (hours) err",
    "Depth (ppm)",
    "Depth (ppm) err",
    "Planet Radius (R_Earth)",
    "Planet Radius (R_Earth) err",
    "Predicted Mass (M_Earth)",
    "Predicted RV Semi-amplitude (m/s)",
    "Stellar Eff Temp (K)",
    "Stellar Radius (R_Sun)",
    "Stellar Mass (M_Sun)",
    "Sectors",
    "Comments",
]


def extract_tic_id(value: str | int | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"(?:TIC\s*)?(\d{4,})", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def fetch_toi_table() -> pd.DataFrame:
    return pd.read_csv(TOI_TABLE_URL, delimiter="|")


def cached_toi_info() -> dict[str, object]:
    if not TOI_CACHE_PATH.exists():
        return {"exists": False, "path": str(TOI_CACHE_PATH)}

    obtained_at = ""
    rows = None
    if TOI_HISTORY_PATH.exists():
        try:
            history = pd.read_csv(TOI_HISTORY_PATH)
        except Exception:
            history = pd.DataFrame()
        if not history.empty:
            latest = history.iloc[-1]
            obtained_at = str(latest.get("obtained_at", ""))
            rows = latest.get("rows", None)

    return {
        "exists": True,
        "path": str(TOI_CACHE_PATH),
        "obtained_at": obtained_at,
        "rows": rows,
    }


def load_toi_table(*, use_cached: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    if use_cached and TOI_CACHE_PATH.exists():
        table = pd.read_csv(TOI_CACHE_PATH)
        info = cached_toi_info()
        info["source"] = "cached"
        return table, info

    table = fetch_toi_table()
    TOI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TOI_CACHE_PATH, index=False)

    obtained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history_row = pd.DataFrame(
        [
            {
                "obtained_at": obtained_at,
                "rows": len(table),
                "columns": len(table.columns),
                "source_url": TOI_TABLE_URL,
                "cache_path": str(TOI_CACHE_PATH),
            }
        ]
    )
    if TOI_HISTORY_PATH.exists():
        try:
            history = pd.read_csv(TOI_HISTORY_PATH)
        except Exception:
            history = pd.DataFrame()
        history = pd.concat([history, history_row], ignore_index=True)
    else:
        history = history_row
    history.to_csv(TOI_HISTORY_PATH, index=False)

    return table, {
        "exists": True,
        "path": str(TOI_CACHE_PATH),
        "obtained_at": obtained_at,
        "rows": len(table),
        "source": "downloaded",
    }


def find_toi_parameters(toi_table: pd.DataFrame, tic_id: str | int) -> pd.DataFrame:
    tic = extract_tic_id(tic_id)
    if not tic or "TIC ID" not in toi_table.columns:
        return pd.DataFrame()

    tic_values = toi_table["TIC ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    matches = toi_table.loc[tic_values == tic].copy()
    if matches.empty:
        return matches
    if "TOI" in matches.columns:
        matches = matches.sort_values("TOI")
    return matches.reset_index(drop=True)


def preferred_toi_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in PREFERRED_TOI_COLUMNS if column in frame.columns]
