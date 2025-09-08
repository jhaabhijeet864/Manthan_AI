import os
import zipfile
import argparse
from pathlib import Path
import pandas as pd

RAW_BASE = Path("data/raw_data")
CMLRE_ZIP = RAW_BASE / "cmlre" / "dwca-vocuherspecimencollections-v1.4.zip"
DWCA_EXTRACT_DIR = RAW_BASE / "cmlre" / "extracted"
WOD_DIR = RAW_BASE / "wod_oceanographic"
PROCESSED_DIR = Path("data/processed_data")
OUTPUT_CSV = PROCESSED_DIR / "fused_occurrence_oceanographic.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "fused_occurrence_oceanographic.parquet"
OCC_ONLY_CSV = PROCESSED_DIR / "occurrence_clean.csv"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DWCA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def extract_dwca(zip_path: Path, extract_dir: Path):
    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)


def load_cmlre_occurrences() -> pd.DataFrame:
    """
    Expects Darwin Core occurrence.txt inside extracted archive.
    """
    extract_dwca(CMLRE_ZIP, DWCA_EXTRACT_DIR)
    occ_file = next((p for p in DWCA_EXTRACT_DIR.rglob("occurrence.txt")), None)
    if occ_file is None:
        raise FileNotFoundError("occurrence.txt not found in Darwin Core archive.")
    # Darwin Core often is tab-delimited
    df = pd.read_csv(occ_file, sep='\t', low_memory=False)
    # Standardize spatial & temporal fields (Darwin Core typical: decimalLatitude, decimalLongitude, eventDate)
    rename_map = {
        'decimalLatitude': 'latitude',
        'decimalLongitude': 'longitude',
        'eventDate': 'event_date',
        'scientificName': 'scientific_name'
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Parse date
    if 'event_date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['event_date'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    # Filter invalid coords
    if {'latitude', 'longitude'}.issubset(df.columns):
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    # Keep essential columns
    keep_cols = [c for c in ['scientific_name', 'latitude', 'longitude', 'timestamp'] if c in df.columns]
    return df[keep_cols].dropna(subset=['latitude', 'longitude']).reset_index(drop=True)


def parse_wod_file(path: Path) -> pd.DataFrame:
    """
    Placeholder parser for WOD profile files (.CTD/.OSD/.XBT).
    Replace with real parsing logic or pre-convert to CSV.
    Currently returns empty DataFrame if not convertible.
    """
    # If you export WOD data to CSV beforehand, detect and load:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        # Stub: attempt fixed-width or skip
        try:
            df = pd.read_fwf(path, nrows=0)  # No spec provided
            df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
    return df


def load_wod_profiles() -> pd.DataFrame:
    """
    Aggregate all WOD files into a unified DataFrame with standardized columns:
      latitude, longitude, timestamp, temperature, salinity, depth
    """
    frames = []
    for f in WOD_DIR.iterdir():
        if f.is_file():
            df = parse_wod_file(f)
            if df.empty:
                continue
            # Attempt column normalization
            rename_map = {
                'lat': 'latitude',
                'lon': 'longitude',
                'long': 'longitude',
                'date': 'timestamp',
                'time': 'time'
            }
            for k, v in rename_map.items():
                if k in df.columns and v not in df.columns:
                    df.rename(columns={k: v}, inplace=True)

            # Combine date+time if separate
            if 'timestamp' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str) + ' ' + df['time'].astype(str),
                                                 errors='coerce')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                df['timestamp'] = pd.NaT

            # Ensure numeric
            for col in ['latitude', 'longitude', 'temperature', 'salinity', 'depth']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            subset_cols = [c for c in ['latitude', 'longitude', 'timestamp', 'temperature', 'salinity', 'depth'] if c in df.columns]
            if subset_cols:
                frames.append(df[subset_cols])
    if not frames:
        return pd.DataFrame(columns=['latitude', 'longitude', 'timestamp', 'temperature', 'salinity', 'depth'])
    wod = pd.concat(frames, ignore_index=True)
    # Basic cleaning (guard: columns guaranteed present due to filtering above)
    if 'latitude' in wod.columns and 'longitude' in wod.columns:
        wod = wod.dropna(subset=['latitude', 'longitude'])
        wod = wod[(wod['latitude'].between(-90, 90)) & (wod['longitude'].between(-180, 180))]
    return wod


def temporal_spatial_merge(occ: pd.DataFrame, ocean: pd.DataFrame,
                           spatial_precision=2, time_freq='D') -> pd.DataFrame:
    """
    Coarse merge: round lat/lon & floor timestamp to day to align.
    Safely handles empty / non-datetime timestamp columns.
    """
    import pandas as pd
    from pandas.api import types as pdt

    for df in (occ, ocean):
        # Timestamp handling
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if pdt.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp_round'] = df['timestamp'].dt.floor(time_freq)
            else:
                df['timestamp_round'] = pd.NaT
        else:
            df['timestamp_round'] = pd.NaT

        # Spatial rounding
        if 'latitude' in df.columns:
            df['lat_round'] = pd.to_numeric(df['latitude'], errors='coerce').round(spatial_precision)
        else:
            df['lat_round'] = float('nan')
        if 'longitude' in df.columns:
            df['lon_round'] = pd.to_numeric(df['longitude'], errors='coerce').round(spatial_precision)
        else:
            df['lon_round'] = float('nan')

    merged = pd.merge(
        occ,
        ocean,
        on=['lat_round', 'lon_round', 'timestamp_round'],
        how='left',
        suffixes=('_occ', '_ocean')
    )
    return merged


def main():
    """Entry point with optional occurrence-only mode.

    --occ-only: skips loading/merging oceanographic data and just outputs a
    cleaned occurrence dataset at OCC_ONLY_CSV.
    """
    parser = argparse.ArgumentParser(description="Data fusion / occurrence preparation")
    parser.add_argument("--occ-only", action="store_true", help="Skip oceanographic merge; output occurrence_clean.csv only")
    args = parser.parse_args()

    ensure_dirs()
    print("Loading CMLRE occurrence data...")
    occ = load_cmlre_occurrences()
    print(f"Occurrences loaded: {len(occ)} rows")

    # Basic occurrence-only cleanup
    occ_clean = occ.drop_duplicates().reset_index(drop=True)
    if args.occ_only:
        print("--occ-only specified: skipping oceanographic data load & fusion.")
        print(f"Saving cleaned occurrences to {OCC_ONLY_CSV}")
        occ_clean.to_csv(OCC_ONLY_CSV, index=False)
        print("Done (occurrence-only mode).")
        return

    print("Loading WOD oceanographic profiles (placeholder)...")
    wod = load_wod_profiles()
    print(f"WOD profiles: {len(wod)} rows")

    if wod.empty:
        print("Warning: WOD profiles DataFrame is empty. Fusion will have NaNs for environmental fields.")

    print("Merging (coarse spatial & daily temporal join)...")
    fused = temporal_spatial_merge(occ_clean, wod)

    print(f"Saving fused dataset to {OUTPUT_CSV} and {OUTPUT_PARQUET}")
    fused.to_csv(OUTPUT_CSV, index=False)
    try:
        fused.to_parquet(OUTPUT_PARQUET, index=False)
    except Exception as e:
        print(f"Parquet save failed (install pyarrow or fastparquet). Error: {e}")

    print("Done.")


if __name__ == "__main__":
    main()