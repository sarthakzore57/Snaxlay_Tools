from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
