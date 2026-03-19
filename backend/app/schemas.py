from typing import Literal

from pydantic import BaseModel, Field


LayoutOption = Literal[2, 4, 6]
DetectionMode = Literal["basic", "smart"]
PaperSize = Literal["A4", "A5", "LETTER"]
JobState = Literal["queued", "running", "completed", "failed"]


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobState
    estimated_processing_seconds: float
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    stage: str
    message: str
    progress_percent: int = 0
    estimated_processing_seconds: float = 0
    actual_processing_seconds: float | None = None
    source_page_count: int | None = None
    extracted_label_count: int | None = None
    output_page_count: int | None = None
    layout: LayoutOption
    detection_mode: DetectionMode
    paper_size: PaperSize
    preview_urls: list[str] = Field(default_factory=list)
    download_url: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    order_count: int = 0


class ProcessResponse(BaseModel):
    job_id: str
    source_page_count: int
    extracted_label_count: int
    output_page_count: int
    layout: LayoutOption
    detection_mode: DetectionMode
    paper_size: PaperSize
    preview_urls: list[str] = Field(default_factory=list)
    download_url: str
    estimated_processing_seconds: float
    actual_processing_seconds: float
    warnings: list[str] = Field(default_factory=list)


class OrderRecordResponse(BaseModel):
    order_key: str
    job_id: str
    source_file: str
    source_page: int
    vendor: str
    platform: str | None = None
    order_id: str | None = None
    suborder_id: str | None = None
    invoice_number: str | None = None
    awb_number: str | None = None
    order_date: str | None = None
    invoice_date: str | None = None
    order_day: str
    customer_name: str | None = None
    city: str | None = None
    state: str | None = None
    product_name: str | None = None
    sku_code: str | None = None
    quantity: int | None = None
    price: float | None = None
    delivery_option: str | None = None


class OrdersSummaryResponse(BaseModel):
    total_orders: int = 0
    total_quantity: int = 0
    total_revenue: float = 0
    cod_orders: int = 0
    prepaid_orders: int = 0


class OrdersDayBucketResponse(BaseModel):
    day: str
    order_count: int
    revenue: float = 0


class OrdersDashboardResponse(BaseModel):
    selected_day: str
    available_days: list[str] = Field(default_factory=list)
    summary: OrdersSummaryResponse
    day_buckets: list[OrdersDayBucketResponse] = Field(default_factory=list)
    orders: list[OrderRecordResponse] = Field(default_factory=list)
