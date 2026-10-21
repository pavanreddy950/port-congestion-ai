"""Shared configuration: paths, ports of interest, and AIS filters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Secrets such as OPENAI_API_KEY live in a local .env file (never committed).
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and not _line.lstrip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # filtered daily AIS parquet files
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

AIS_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"

# Bounding boxes (lat_min, lat_max, lon_min, lon_max) covering each port's
# terminals, approach channel and offshore anchorages where ships queue.
# lat/lon is the weather point; wave_lat/wave_lon is an offshore point for
# wave height (the marine model has no values over land or inside harbours).
PORTS = {
    "LA_LongBeach": {"bbox": (33.40, 33.80, -118.50, -117.90), "lat": 33.73, "lon": -118.22,
                     "wave_lat": 33.65, "wave_lon": -118.20},
    "Houston":      {"bbox": (28.80, 29.80, -95.30, -94.10),  "lat": 29.35, "lon": -94.80,
                     "wave_lat": 29.20, "wave_lon": -94.60},
    "NY_NJ":        {"bbox": (40.40, 40.75, -74.30, -73.85),  "lat": 40.62, "lon": -74.05,
                     "wave_lat": 40.45, "wave_lon": -73.90},
    "Savannah":     {"bbox": (31.75, 32.15, -81.20, -80.20),  "lat": 32.02, "lon": -80.90,
                     "wave_lat": 31.95, "wave_lon": -80.65},
}

# AIS vessel type codes: 70-79 cargo, 80-89 tanker. Tugs, ferries, fishing
# and pleasure craft do not queue for berths, so they are excluded.
VESSEL_TYPE_RANGE = (70, 89)

AIS_COLUMNS = [
    "MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading",
    "VesselName", "IMO", "VesselType", "Status", "Length", "Width", "Draft",
]
