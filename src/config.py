# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TELCO_CSV = RAW_DIR / "telco.csv"
SAAS_CSV = RAW_DIR / "saas.csv"
BANKCHURNERS_CSV = RAW_DIR / "bankchurners.csv"
