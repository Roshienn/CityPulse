"""Dataset loading and cleaning utilities for the Beijing PRSA dataset."""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

NUMERIC_COLUMNS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
]
REQUIRED_COLUMNS = ["year", "month", "day", "hour", "station", "PM2.5"]


def discover_files(data_dir: Path) -> List[Path]:
    """Return station files, excluding generated cleaned outputs."""
    return sorted(
        p for p in Path(data_dir).glob("*.csv")
        if not p.name.startswith("cleaned_") and not p.name.startswith("benchmark_")
    )


def _clean_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")

    frame = frame.copy()
    frame["station"] = frame["station"].fillna(source.stem.split("_")[2])
    frame["datetime"] = pd.to_datetime(
        frame[["year", "month", "day", "hour"]], errors="coerce"
    )
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].interpolate(limit_direction="both")
    return frame.reset_index(drop=True)


def load_dataset(
    data_dir: Path = Path("datasets"),
    cleaned_path: Optional[Path] = None,
    save_cleaned: bool = True,
) -> pd.DataFrame:
    """Load all station files, clean values, and optionally save a master CSV."""
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No station CSV files found in {Path(data_dir).resolve()}")
    frames = [_clean_frame(pd.read_csv(path), path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["datetime", "station"]).reset_index(drop=True)
    if save_cleaned:
        output = cleaned_path or Path(data_dir) / "cleaned_citypulse.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
    return data


def load_or_create_cleaned(
    data_dir: Path = Path("datasets"),
    cleaned_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Use a cleaned dataset when it is newer than its source files."""
    output = cleaned_path or Path(data_dir) / "cleaned_citypulse.csv"
    files = discover_files(data_dir)
    if output.exists() and files and output.stat().st_mtime >= max(p.stat().st_mtime for p in files):
        data = pd.read_csv(output, parse_dates=["datetime"])
        return data
    return load_dataset(data_dir, output, save_cleaned=True)

