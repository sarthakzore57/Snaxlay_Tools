from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
ORDERS_FILE = STORAGE_DIR / "orders.jsonl"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
ORDERS_FILE.touch(exist_ok=True)
