"""MAST query helpers for TESS photometry discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from astroquery.mast import Observations


TIC_RE = re.compile(r"^(?:TIC\s*)?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class MastQueryResult:
    target: str
    resolved_target: str
    observations: pd.DataFrame
    products: pd.DataFrame


def normalize_target_name(target: str) -> str:
    """Return the best MAST target string for a user-entered target."""
    cleaned = " ".join(target.strip().split())
    match = TIC_RE.match(cleaned)
    if match:
        return match.group(1)
    return cleaned


def _find_tic_from_simbad(target: str) -> str | None:
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        return None

    simbad = Simbad()
    simbad.add_votable_fields("ids")
    result = simbad.query_object(target)
    id_column = _column_name(result, "ids") if result is not None else None
    if result is None or len(result) == 0 or id_column is None:
        return None

    ids = str(result[id_column][0]).split("|")
    for identifier in ids:
        match = re.search(r"\bTIC\s+(\d+)\b", identifier, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _column_name(table, name: str) -> str | None:
    if table is None:
        return None
    for column in table.colnames:
        if column.lower() == name.lower():
            return column
    return None


def _to_dataframe(table) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    return table.to_pandas()


def _filter_observations(observations: pd.DataFrame, cadence: str) -> pd.DataFrame:
    if observations.empty:
        return observations

    filtered = observations.copy()
    filtered["_mast_row"] = filtered.index
    if "obs_collection" in filtered:
        filtered = filtered[filtered["obs_collection"].isin(["TESS", "HLSP"])]
    if "dataproduct_type" in filtered:
        filtered = filtered[filtered["dataproduct_type"].fillna("").str.lower().eq("timeseries")]

    if cadence != "All available cadences" and "t_exptime" in filtered:
        exposure = pd.to_numeric(filtered["t_exptime"], errors="coerce")
        if cadence == "20 s":
            mask = exposure.between(15, 25)
        elif cadence == "2 min":
            mask = exposure.between(100, 140)
        elif cadence == "10 min":
            mask = exposure.between(580, 620)
        elif cadence == "30 min":
            mask = exposure.between(1700, 1900)
        else:
            mask = pd.Series(True, index=filtered.index)
        filtered = filtered[mask]

    sort_cols = [col for col in ["sequence_number", "t_min", "obs_id"] if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols)
    return filtered.reset_index(drop=True)


def _product_summary(observation_table) -> pd.DataFrame:
    if observation_table is None or len(observation_table) == 0:
        return pd.DataFrame()

    products = Observations.get_product_list(observation_table)
    products_df = _to_dataframe(products)
    if products_df.empty:
        return products_df

    keep = [
        col
        for col in [
            "obs_id",
            "productFilename",
            "description",
            "productType",
            "productSubGroupDescription",
            "size",
            "dataURI",
        ]
        if col in products_df.columns
    ]
    products_df = products_df[keep]
    if "description" in products_df:
        light_curve_mask = products_df["description"].fillna("").str.contains("Light curves|Lightcurve|FITS", case=False)
        products_df = products_df[light_curve_mask]
    return products_df.reset_index(drop=True)


def _query_by_target_name(target: str):
    return Observations.query_criteria(
        obs_collection=["TESS", "HLSP"],
        target_name=target,
        dataproduct_type="timeseries",
    )


def query_tess_photometry(target: str, cadence: str = "2 min") -> MastQueryResult:
    """Resolve the input to a TIC ID via SIMBAD, then query MAST TESS products."""
    normalized = normalize_target_name(target)
    if TIC_RE.match(target.strip()):
        query_target = normalized
    else:
        tic = _find_tic_from_simbad(normalized)
        if not tic:
            raise ValueError(
                f"Could not resolve `{target}` to a TIC ID via SIMBAD. "
                "Try entering a TIC ID directly, for example `TIC 243921117`."
            )
        query_target = tic

    observation_table = _query_by_target_name(query_target)
    observations = _filter_observations(_to_dataframe(observation_table), cadence)

    if observations.empty:
        products = pd.DataFrame()
    else:
        row_indices = observations["_mast_row"].tolist() if "_mast_row" in observations else observations.index.tolist()
        observation_subset = observation_table[row_indices] if len(observation_table) >= len(row_indices) else observation_table
        products = _product_summary(observation_subset)

    return MastQueryResult(
        target=target,
        resolved_target=query_target,
        observations=observations.drop(columns=["_mast_row"], errors="ignore"),
        products=products,
    )


def safe_target_slug(target: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.strip())
    return slug.strip("_") or "target"


def download_tess_products(
    target: str,
    cadence: str,
    product_filenames: list[str],
    download_root: str | Path,
) -> list[Path]:
    """Download selected MAST products and return local file paths."""
    if not product_filenames:
        return []

    result = query_tess_photometry(target, cadence)
    if result.observations.empty:
        return []

    observation_table = _query_by_target_name(result.resolved_target)
    products = Observations.get_product_list(observation_table)
    product_df = products.to_pandas()
    selected_mask = product_df["productFilename"].isin(product_filenames)
    selected_products = products[selected_mask.to_numpy()]

    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    manifest = Observations.download_products(selected_products, download_dir=str(download_root))
    if manifest is None or len(manifest) == 0 or "Local Path" not in manifest.colnames:
        return []
    return [Path(path) for path in manifest["Local Path"] if path]
