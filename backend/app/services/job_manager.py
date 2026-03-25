from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import ORDERS_FILE
from app.services.pdf_processor import PDFProcessor, ProcessResult


@dataclass
class JobRecord:
    job_id: str
    filename: str
    layout: int
    detection_mode: str
    paper_size: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start."
    progress_percent: int = 0
    estimated_processing_seconds: float = 0
    actual_processing_seconds: float | None = None
    source_page_count: int | None = None
    extracted_label_count: int | None = None
    output_page_count: int | None = None
    preview_urls: list[str] = field(default_factory=list)
    download_url: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    order_count: int = 0
    orders: list[dict] = field(default_factory=list)


class JobManager:
    def __init__(self, processor: PDFProcessor | None = None) -> None:
        self.processor = processor or PDFProcessor()
        self._jobs: dict[str, JobRecord] = {}
        self._orders_file: Path = ORDERS_FILE
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def create_job(
        self,
        documents: list[tuple[bytes, str]],
        layout: int,
        detection_mode: str,
        paper_size: str,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        estimate = self._initial_estimate(
            pdf_size_bytes=sum(len(pdf_bytes) for pdf_bytes, _ in documents),
            detection_mode=detection_mode,
        )
        record = JobRecord(
            job_id=job_id,
            filename=", ".join(filename for _, filename in documents),
            layout=layout,
            detection_mode=detection_mode,
            paper_size=paper_size,
            estimated_processing_seconds=estimate,
            message="Queued for processing.",
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run_job, job_id, documents, layout, detection_mode, paper_size)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_orders(self, selected_day: str = "today") -> dict[str, object]:
        today = datetime.now().date().isoformat()
        normalized_day = today if selected_day == "today" else selected_day
        all_orders = self._load_orders()

        available_days = sorted({self._dashboard_day(order) for order in all_orders if self._dashboard_day(order)}, reverse=True)
        filtered_orders = [
            order for order in all_orders if self._dashboard_day(order) == normalized_day
        ] if normalized_day else all_orders
        filtered_orders.sort(
            key=lambda order: (
                self._dashboard_day(order) or "",
                order.get("order_date") or "",
                order.get("source_file", ""),
                order.get("source_page", 0),
            ),
            reverse=True,
        )

        total_quantity = sum(order.get("quantity") or 0 for order in filtered_orders)
        total_revenue = round(sum(order.get("price") or 0 for order in filtered_orders), 2)
        cod_orders = sum(1 for order in filtered_orders if (order.get("delivery_option") or "").upper() == "COD")
        prepaid_orders = sum(1 for order in filtered_orders if (order.get("delivery_option") or "").upper() == "PREPAID")

        buckets: dict[str, dict[str, float | int | str]] = {}
        for order in all_orders:
            day = self._dashboard_day(order)
            if not day:
                continue
            bucket = buckets.setdefault(day, {"day": day, "order_count": 0, "revenue": 0.0})
            bucket["order_count"] += 1
            bucket["revenue"] = round(float(bucket["revenue"]) + float(order.get("price") or 0), 2)

        day_buckets = sorted(buckets.values(), key=lambda bucket: bucket["day"], reverse=True)
        return {
            "selected_day": selected_day,
            "available_days": available_days,
            "summary": {
                "total_orders": len(filtered_orders),
                "total_quantity": total_quantity,
                "total_revenue": total_revenue,
                "cod_orders": cod_orders,
                "prepaid_orders": prepaid_orders,
            },
            "day_buckets": day_buckets,
            "orders": filtered_orders,
        }

    def _order_identity(self, order: dict) -> str:
        order_id = (order.get("order_id") or "").strip().upper()
        if order_id:
            return f"order:{order_id}"
        return str(order.get("dedupe_key") or order.get("order_key"))

    def _dashboard_day(self, order: dict) -> str | None:
        return order.get("order_date") or order.get("invoice_date") or order.get("order_day")

    def _run_job(
        self,
        job_id: str,
        documents: list[tuple[bytes, str]],
        layout: int,
        detection_mode: str,
        paper_size: str,
    ) -> None:
        started_at = time.perf_counter()
        self._update_job(
            job_id,
            status="running",
            stage="starting",
            message="Preparing the PDF job.",
            progress_percent=3,
        )
        try:
            result = self.processor.process_pdfs(
                documents=documents,
                layout=layout,
                detection_mode=detection_mode,
                paper_size=paper_size,
                progress_callback=lambda stage, message, progress, meta=None: self._handle_progress(
                    job_id, stage, message, progress, meta or {}
                ),
            )
            self._apply_result(job_id, result, round(time.perf_counter() - started_at, 2))
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                message="Processing failed.",
                progress_percent=100,
                error=str(exc),
                actual_processing_seconds=round(time.perf_counter() - started_at, 2),
            )

    def _handle_progress(self, job_id: str, stage: str, message: str, progress: int, meta: dict) -> None:
        payload = {
            "stage": stage,
            "message": message,
            "progress_percent": progress,
        }
        if "estimated_processing_seconds" in meta:
            payload["estimated_processing_seconds"] = meta["estimated_processing_seconds"]
        if "source_page_count" in meta:
            payload["source_page_count"] = meta["source_page_count"]
        if "warnings" in meta:
            payload["warnings"] = meta["warnings"]
        self._update_job(job_id, **payload)

    def _apply_result(self, job_id: str, result: ProcessResult, elapsed: float) -> None:
        deduped_orders_map: dict[str, dict] = {}
        for order in result.orders:
            normalized_order = {
                **order,
                "job_id": job_id,
            }
            deduped_orders_map[self._order_identity(normalized_order)] = normalized_order
        normalized_orders = list(deduped_orders_map.values())

        self._save_orders_for_job(job_id, normalized_orders)
        self._update_job(
            job_id,
            status="completed",
            stage="completed",
            message="Print layout is ready.",
            progress_percent=100,
            estimated_processing_seconds=result.estimated_processing_seconds,
            actual_processing_seconds=elapsed,
            source_page_count=result.source_page_count,
            extracted_label_count=result.extracted_label_count,
            output_page_count=result.output_page_count,
            preview_urls=result.preview_urls,
            download_url=result.download_url,
            warnings=result.warnings,
            order_count=len(normalized_orders),
            orders=normalized_orders,
        )

    def _update_job(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)

    def _load_orders(self) -> list[dict]:
        with self._lock:
            if not self._orders_file.exists():
                return []

            orders: list[dict] = []
            for raw_line in self._orders_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    orders.append(payload)
            return orders

    def _save_orders_for_job(self, job_id: str, orders: list[dict]) -> None:
        with self._lock:
            existing_orders = self._load_orders_unlocked()
            dedupe_keys = {self._order_identity(order) for order in orders}
            filtered_orders = [
                order
                for order in existing_orders
                if order.get("job_id") != job_id and self._order_identity(order) not in dedupe_keys
            ]
            filtered_orders.extend(orders)
            serialized = "\n".join(json.dumps(order, ensure_ascii=True) for order in filtered_orders)
            if serialized:
                serialized += "\n"
            self._orders_file.write_text(serialized, encoding="utf-8")

    def _load_orders_unlocked(self) -> list[dict]:
        if not self._orders_file.exists():
            return []

        orders: list[dict] = []
        for raw_line in self._orders_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                orders.append(payload)
        return orders

    def _initial_estimate(self, pdf_size_bytes: int, detection_mode: str) -> float:
        size_mb = pdf_size_bytes / (1024 * 1024)
        base = 2.2 if detection_mode == "basic" else 4.0
        return round(max(2.0, base + (size_mb * 3.2)), 1)
