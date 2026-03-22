import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import JOBS_DIR
from app.schemas import JobStatusResponse, JobSubmitResponse, OrdersDashboardResponse
from app.services.job_manager import JobManager


app = FastAPI(title="Shipping Label Arranger", version="1.0.0")
job_manager = JobManager()

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
extra_cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
cors_origins.extend(extra_cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://([a-zA-Z0-9-]+\.)?(vercel\.app|netlify\.app|onrender\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Shipping Label Arranger API is running.",
        "health": "/api/health",
        "process": "/api/process",
    }


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process", response_model=JobSubmitResponse)
async def process_pdf(
    files: list[UploadFile] | None = File(None),
    file_items: list[UploadFile] | None = File(None, alias="file"),
    layout: int = Form(2),
    detection_mode: str = Form("basic"),
    paper_size: str = Form("A4"),
) -> JobSubmitResponse:
    files = [*(files or []), *(file_items or [])]
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if layout not in {2, 4, 6}:
        raise HTTPException(status_code=400, detail="Layout must be 2, 4, or 6.")
    if detection_mode not in {"basic", "smart"}:
        raise HTTPException(status_code=400, detail="Detection mode must be 'basic' or 'smart'.")
    if paper_size not in {"A4", "A5", "LETTER"}:
        raise HTTPException(status_code=400, detail="Unsupported paper size.")

    documents: list[tuple[bytes, str]] = []
    for file in files:
        if file.content_type not in {"application/pdf", "application/x-pdf"} and not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"Uploaded PDF is empty: {file.filename}")
        documents.append((payload, file.filename or "upload.pdf"))

    record = job_manager.create_job(
        documents=documents,
        layout=layout,
        detection_mode=detection_mode,
        paper_size=paper_size,
    )
    return JobSubmitResponse(
        job_id=record.job_id,
        status=record.status,
        estimated_processing_seconds=record.estimated_processing_seconds,
        status_url=f"/api/jobs/{record.job_id}",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        message=job.message,
        progress_percent=job.progress_percent,
        estimated_processing_seconds=job.estimated_processing_seconds,
        actual_processing_seconds=job.actual_processing_seconds,
        source_page_count=job.source_page_count,
        extracted_label_count=job.extracted_label_count,
        output_page_count=job.output_page_count,
        layout=job.layout,
        detection_mode=job.detection_mode,
        paper_size=job.paper_size,
        preview_urls=job.preview_urls,
        download_url=job.download_url,
        warnings=job.warnings,
        error=job.error,
        order_count=job.order_count,
    )


@app.get("/api/orders", response_model=OrdersDashboardResponse)
def get_orders(day: str = "today") -> OrdersDashboardResponse:
    return OrdersDashboardResponse(**job_manager.list_orders(day))


@app.get("/api/jobs/{job_id}/preview/{image_name}")
def get_preview(job_id: str, image_name: str) -> FileResponse:
    preview_path = JOBS_DIR / job_id / "preview" / image_name
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview image not found.")
    return FileResponse(preview_path)


@app.get("/api/jobs/{job_id}/download")
def download_pdf(job_id: str) -> FileResponse:
    pdf_path = JOBS_DIR / job_id / "arranged-labels.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Generated PDF not found.")
    return FileResponse(pdf_path, media_type="application/pdf", filename="arranged-labels.pdf")
