from __future__ import annotations

import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A4, A5, LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.config import JOBS_DIR


PAGE_SIZES = {
    "A4": A4,
    "A5": A5,
    "LETTER": LETTER,
}

SOURCE_RENDER_DPI_BASIC = 150
SOURCE_RENDER_DPI_SMART = 180
PREVIEW_IMAGE_WIDTH = 900
SMART_DETECTION_MAX_DIMENSION = 1800
OUTPUT_TARGET_DPI = 240


@dataclass
class ProcessResult:
    job_id: str
    source_page_count: int
    extracted_label_count: int
    output_page_count: int
    preview_urls: list[str]
    download_url: str
    estimated_processing_seconds: float
    actual_processing_seconds: float
    warnings: list[str]
    orders: list[dict[str, object]]


class PDFProcessor:
    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root or JOBS_DIR
        self.max_workers = max(2, min(8, (os.cpu_count() or 4)))

    def process_pdfs(
        self,
        documents: list[tuple[bytes, str]],
        layout: int,
        detection_mode: str,
        paper_size: str,
        progress_callback=None,
    ) -> ProcessResult:
        started_at = time.perf_counter()
        job_id = uuid.uuid4().hex
        job_dir = self.output_root / job_id
        preview_dir = job_dir / "preview"
        for directory in (job_dir, preview_dir):
            directory.mkdir(parents=True, exist_ok=True)

        source_render_dpi = SOURCE_RENDER_DPI_SMART if detection_mode == "smart" else SOURCE_RENDER_DPI_BASIC
        self._notify(progress_callback, "rendering", "Rendering PDF pages into working images.", 10)
        source_pages: list[dict[str, object]] = []
        total_input_bytes = 0
        for document_index, (pdf_bytes, filename) in enumerate(documents, start=1):
            source_pdf_path = job_dir / f"{document_index:02d}-{self._clean_filename(filename)}"
            source_pdf_path.write_bytes(pdf_bytes)
            total_input_bytes += len(pdf_bytes)
            source_pages.extend(self._load_pdf_pages(pdf_bytes, dpi=source_render_dpi, source_file=filename))

        estimated_processing_seconds = self._estimate_processing_seconds(
            source_page_count=len(source_pages),
            pdf_size_bytes=total_input_bytes,
            detection_mode=detection_mode,
        )
        self._notify(
            progress_callback,
            "rendered",
            f"Rendered {len(source_pages)} page(s). Starting label extraction.",
            20,
            {
                "source_page_count": len(source_pages),
                "estimated_processing_seconds": estimated_processing_seconds,
            },
        )

        page_inputs = [
            (page_index, page_payload["image"], page_payload["text_label_boxes"], layout, detection_mode)
            for page_index, page_payload in enumerate(source_pages, start=1)
        ]
        page_results: list[dict[str, list]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(page_inputs)))) as executor:
            futures = [executor.submit(self._process_page, page_input) for page_input in page_inputs]
            for completed_count, future in enumerate(as_completed(futures), start=1):
                page_results.append(future.result())
                extraction_progress = 20 + int((completed_count / max(1, len(page_inputs))) * 45)
                self._notify(
                    progress_callback,
                    "extracting",
                    f"Extracted labels from {completed_count} of {len(page_inputs)} page(s).",
                    extraction_progress,
                    {
                        "source_page_count": len(page_inputs),
                    },
                )

        warnings: list[str] = []
        extracted_labels: list[Image.Image] = []
        extracted_orders: list[dict[str, object]] = []
        for source_page in source_pages:
            extracted_orders.extend(source_page.get("orders", []))
        for page_result in page_results:
            warnings.extend(page_result["warnings"])
            extracted_labels.extend(page_result["labels"])

        output_pdf_path = job_dir / "arranged-labels.pdf"
        self._notify(
            progress_callback,
            "layout",
            f"Arranging {len(extracted_labels)} label(s) on {paper_size} sheets.",
            72,
            {
                "warnings": warnings,
            },
        )
        output_page_count, preview_images = self._create_output_pdf(
            labels=extracted_labels,
            output_pdf_path=output_pdf_path,
            layout=layout,
            paper_size=paper_size,
        )

        preview_urls: list[str] = []
        self._notify(progress_callback, "preview", "Encoding preview images for the dashboard.", 88)
        for index, image in enumerate(preview_images, start=1):
            preview_name = f"preview-{index:03d}.jpg"
            image.save(preview_dir / preview_name, "JPEG", quality=82, optimize=True)
            preview_urls.append(f"/api/jobs/{job_id}/preview/{preview_name}")

        actual_processing_seconds = round(time.perf_counter() - started_at, 2)
        self._notify(progress_callback, "completed", "Layout generation finished.", 100)

        return ProcessResult(
            job_id=job_id,
            source_page_count=len(source_pages),
            extracted_label_count=len(extracted_labels),
            output_page_count=output_page_count,
            preview_urls=preview_urls,
            download_url=f"/api/jobs/{job_id}/download",
            estimated_processing_seconds=estimated_processing_seconds,
            actual_processing_seconds=actual_processing_seconds,
            warnings=warnings,
            orders=extracted_orders,
        )

    def _process_page(self, page_input: tuple[int, Image.Image, list[tuple[int, int, int, int]], int, str]) -> dict[str, list]:
        page_index, page_image, text_label_boxes, layout, detection_mode = page_input
        warnings: list[str] = []
        labels = self._extract_labels(
            page_image,
            layout=layout,
            detection_mode=detection_mode,
            text_label_boxes=text_label_boxes,
        )
        if detection_mode == "smart" and not labels:
            warnings.append(f"Smart detection found no labels on page {page_index}; basic grid split was used.")
            labels = self._extract_labels(
                page_image,
                layout=layout,
                detection_mode="basic",
                text_label_boxes=text_label_boxes,
            )

        normalized_labels = [self._normalize_label(self._remove_invoice_panel(label, layout)) for label in labels]
        return {
            "labels": normalized_labels,
            "warnings": warnings,
        }

    def _remove_invoice_panel(self, label: Image.Image, layout: int) -> Image.Image:
        width, height = label.size
        if width < height * 1.1:
            return label

        rgb = np.array(label.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        content_box = self._find_invoice_free_area(merged, width, height, layout)
        if content_box is None:
            return label

        left, top, crop_width, crop_height = content_box
        return label.crop((left, top, left + crop_width, top + crop_height))

    def _estimate_processing_seconds(self, source_page_count: int, pdf_size_bytes: int, detection_mode: str) -> float:
        size_mb = pdf_size_bytes / (1024 * 1024)
        per_page_seconds = 0.85 if detection_mode == "basic" else 1.45
        worker_gain = max(1.0, min(self.max_workers, max(1, source_page_count)) * 0.55)
        estimate = 1.2 + ((source_page_count * per_page_seconds) + (size_mb * 0.6)) / worker_gain
        return round(max(1.0, estimate), 1)

    def _notify(self, callback, stage: str, message: str, progress: int, meta: dict | None = None) -> None:
        if callback:
            callback(stage, message, progress, meta or {})

    def _load_pdf_pages(self, pdf_bytes: bytes, dpi: int, source_file: str) -> list[dict[str, object]]:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return self._render_document(document, dpi, source_file)
        finally:
            document.close()

    def _render_document(self, document: fitz.Document, dpi: int, source_file: str) -> list[dict[str, object]]:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        pages: list[dict[str, object]] = []
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append(
                {
                    "image": image,
                    "text_label_boxes": self._find_vector_label_boxes(page, scale) or self._find_text_label_boxes(page, scale),
                    "orders": self._extract_orders_from_page(page, source_file, page_number),
                }
            )
        return pages

    def _extract_orders_from_page(self, page: fitz.Page, source_file: str, page_number: int) -> list[dict[str, object]]:
        text = page.get_text("text")
        if not text.strip():
            return []

        vendor = self._detect_vendor(text)
        if vendor == "Snapdeal":
            order = self._parse_snapdeal_order(text, source_file, page_number)
        elif vendor in {"Flipkart", "Shopsy"}:
            order = self._parse_flipkart_order(text, source_file, page_number, vendor)
        else:
            order = self._parse_generic_order(text, source_file, page_number, vendor)

        return [order] if order else []

    def _detect_vendor(self, text: str) -> str:
        upper_text = text.upper()
        if "TAX INVOICE (ORIGINAL FOR RECIPIENT)" in upper_text and "SNAPDEAL" in upper_text:
            return "Snapdeal"
        if "FLIPKART" in upper_text or "E-KART LOGISTICS" in upper_text:
            return "Flipkart"
        if "SHOPSY" in upper_text:
            return "Shopsy"
        return "Unknown"

    def _parse_snapdeal_order(self, text: str, source_file: str, page_number: int) -> dict[str, object] | None:
        product_name = self._extract_snapdeal_product_name(text)
        customer_name = self._first_capture(text, r"DELIVERY ADDRESS\s*([^\n]+)")
        city_state = self._first_capture(text, r"CITY/STATE\s*([^\n]+)")
        city, state = self._split_city_state(city_state)
        order_date = self._normalize_date(self._first_capture(text, r"ORDER DATE\s*:\s*([^\n]+)"))
        invoice_date = self._normalize_date(self._first_capture(text, r"INVOICE DATE\s*:\s*([^\n]+)"))
        price = self._to_float(self._first_capture(text, r"TOTAL\(\s*INCLUSIVE OF TAXES\)\s*Rs\.?\s*([0-9]+(?:\.[0-9]+)?)"))
        order_day = order_date or invoice_date or datetime.now().date().isoformat()
        order_id = self._first_capture(text, r"ORDER NO\.\s*:\s*([A-Z0-9]+)")
        suborder_id = self._first_capture(text, r"SUBORDER NO\.\s*:\s*([A-Z0-9]+)") or self._first_capture(
            text, r"SUBORDER CODE\s*([0-9A-Z]+)"
        )
        quantity = self._extract_snapdeal_quantity(text, suborder_id)
        sku_code = self._first_capture(text, r"SKU CODE:\s*([A-Z0-9\-]+)") or self._extract_sku_from_suborder_line(text)
        delivery_option = self._first_capture(text, r"\b(COD|PREPAID)\b")
        awb_number = self._first_capture(text, r"Snapdeal Reference No\s*([A-Z0-9]+)")
        invoice_number = self._first_capture(text, r"INVOICE NUMBER\s*:\s*([A-Z0-9/\-]+)")

        return self._build_order_record(
            source_file=source_file,
            page_number=page_number,
            vendor="Snapdeal",
            platform="Snapdeal",
            order_id=order_id,
            suborder_id=suborder_id,
            invoice_number=invoice_number,
            awb_number=awb_number,
            order_date=order_date,
            invoice_date=invoice_date,
            order_day=order_day,
            customer_name=customer_name,
            city=city,
            state=state,
            product_name=product_name,
            sku_code=sku_code,
            quantity=quantity,
            price=price,
            delivery_option=delivery_option,
        )

    def _parse_flipkart_order(
        self, text: str, source_file: str, page_number: int, vendor: str
    ) -> dict[str, object] | None:
        sku_code, product_name = self._extract_flipkart_product_data(text)
        customer_name = self._first_capture(text, r"Shipping/Customer address:\s*Name:\s*([^,\n]+)")
        address_block = self._first_capture(text, r"Shipping/Customer address:\s*Name:.*?(.*?)(?:Not for resale\.|Printed at)", flags=re.I | re.S)
        city = self._extract_city_from_address(address_block)
        order_date = self._normalize_date(self._first_capture(text, r"Order Date:\s*([^\n]+)"))
        invoice_date = self._normalize_date(self._first_capture(text, r"Invoice Date:\s*([^\n]+(?:\n[AP]M)?)"))
        price = self._to_float(self._first_capture(text, r"TOTAL PRICE:\s*([0-9]+(?:\.[0-9]+)?)"))
        quantity = self._to_int(self._first_capture(text, r"TOTAL QTY:\s*([0-9]+)")) or self._to_int(
            self._first_capture(text, r"\bQTY\b\s*.*?\n([0-9]+)\n", flags=re.I | re.S)
        )
        order_day = order_date or invoice_date or datetime.now().date().isoformat()
        order_id = self._first_capture(text, r"Order Id:\s*([A-Z0-9]+)")
        awb_number = self._first_capture(text, r"AWB No\.\s*([A-Z0-9]+)")
        invoice_number = self._first_capture(text, r"Invoice No:\s*([A-Z0-9]+)")
        delivery_option = self._first_capture(text, r"\b(COD|PREPAID)\b")

        return self._build_order_record(
            source_file=source_file,
            page_number=page_number,
            vendor=vendor,
            platform=vendor,
            order_id=order_id,
            suborder_id=None,
            invoice_number=invoice_number,
            awb_number=awb_number,
            order_date=order_date,
            invoice_date=invoice_date,
            order_day=order_day,
            customer_name=customer_name,
            city=city,
            state=None,
            product_name=product_name,
            sku_code=sku_code,
            quantity=quantity,
            price=price,
            delivery_option=delivery_option,
        )

    def _parse_generic_order(
        self, text: str, source_file: str, page_number: int, vendor: str
    ) -> dict[str, object] | None:
        order_date = self._normalize_date(
            self._first_capture(text, r"(?:ORDER DATE|Order Date)\s*:?\s*([^\n]+)")
            or self._first_capture(text, r"(?:INVOICE DATE|Invoice Date)\s*:?\s*([^\n]+)")
        )
        order_day = order_date or datetime.now().date().isoformat()
        return self._build_order_record(
            source_file=source_file,
            page_number=page_number,
            vendor=vendor,
            platform=vendor if vendor != "Unknown" else None,
            order_id=self._first_capture(text, r"(?:ORDER NO\.|Order Id)\s*:?\s*([A-Z0-9]+)"),
            suborder_id=None,
            invoice_number=self._first_capture(text, r"(?:INVOICE NUMBER|Invoice No)\s*:?\s*([A-Z0-9/\-]+)"),
            awb_number=self._first_capture(text, r"AWB No\.\s*([A-Z0-9]+)"),
            order_date=order_date,
            invoice_date=None,
            order_day=order_day,
            customer_name=self._first_capture(text, r"DELIVERY ADDRESS\s*([^\n]+)"),
            city=None,
            state=None,
            product_name=None,
            sku_code=None,
            quantity=self._to_int(self._first_capture(text, r"QUANTITY\s*([0-9]+)")),
            price=self._to_float(self._first_capture(text, r"TOTAL PRICE:\s*([0-9]+(?:\.[0-9]+)?)")),
            delivery_option=self._first_capture(text, r"\b(COD|PREPAID)\b"),
        )

    def _build_order_record(
        self,
        *,
        source_file: str,
        page_number: int,
        vendor: str,
        platform: str | None,
        order_id: str | None,
        suborder_id: str | None,
        invoice_number: str | None,
        awb_number: str | None,
        order_date: str | None,
        invoice_date: str | None,
        order_day: str,
        customer_name: str | None,
        city: str | None,
        state: str | None,
        product_name: str | None,
        sku_code: str | None,
        quantity: int | None,
        price: float | None,
        delivery_option: str | None,
    ) -> dict[str, object]:
        return {
            "order_key": f"{self._clean_filename(source_file)}-{page_number}-{order_id or invoice_number or awb_number or vendor}",
            "dedupe_key": f"order:{order_id.upper()}" if order_id else f"fallback:{self._clean_filename(source_file)}:{page_number}:{invoice_number or awb_number or vendor}",
            "source_file": source_file,
            "source_page": page_number,
            "vendor": vendor,
            "platform": platform,
            "order_id": order_id,
            "suborder_id": suborder_id,
            "invoice_number": invoice_number,
            "awb_number": awb_number,
            "order_date": order_date,
            "invoice_date": invoice_date,
            "order_day": order_day,
            "customer_name": customer_name,
            "city": city,
            "state": state,
            "product_name": product_name,
            "sku_code": sku_code,
            "quantity": quantity,
            "price": price,
            "delivery_option": delivery_option.upper() if delivery_option else None,
        }

    def _extract_snapdeal_product_name(self, text: str) -> str | None:
        product_block = re.search(r"PRODUCT NAME\s*QUANTITY\s*(.*?)\s*AMOUNT ALREADY PAID", text, re.I | re.S)
        if product_block:
            lines = [line.strip() for line in product_block.group(1).splitlines() if line.strip()]
            if lines:
                first_line = lines[0]
                if not re.fullmatch(r"[A-Z0-9\-]+", first_line, re.I) and first_line.upper() != "QUANTITY":
                    return first_line

        match = re.search(r"ITEM DESCRIPTION\s*(.*?)\s*SKU CODE:", text, re.I | re.S)
        if not match:
            return None
        lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        for line in lines:
            upper_line = line.upper()
            if "ITEM DESCRIPTION" in upper_line or "SKU CODE:" in upper_line or upper_line in {"QTY", "RATE", "TOTAL", "DISC"}:
                continue
            return line.title() if line.isupper() else line
        return None

    def _extract_snapdeal_quantity(self, text: str, suborder_id: str | None) -> int | None:
        product_block = re.search(r"PRODUCT NAME\s*QUANTITY\s*(.*?)\s*AMOUNT ALREADY PAID", text, re.I | re.S)
        if product_block:
            lines = [line.strip() for line in product_block.group(1).splitlines() if line.strip()]
            for line in reversed(lines):
                quantity = self._safe_quantity(line, suborder_id)
                if quantity is not None:
                    return quantity

        invoice_block = re.search(r"HSN:[^\n]*\n([^\n]+)", text, re.I)
        if invoice_block:
            quantity = self._safe_quantity(invoice_block.group(1), suborder_id)
            if quantity is not None:
                return quantity

        quantity = self._safe_quantity(self._first_capture(text, r"QUANTITY\s*([0-9]+)"), suborder_id)
        if quantity is not None:
            return quantity
        return 1

    def _safe_quantity(self, value: str | None, suborder_id: str | None) -> int | None:
        quantity = self._to_int(value)
        if quantity is None:
            return None
        if suborder_id and str(quantity) == str(suborder_id):
            return None
        if quantity <= 0 or quantity > 50:
            return None
        return quantity

    def _extract_flipkart_product_data(self, text: str) -> tuple[str | None, str | None]:
        match = re.search(r"SKU ID\s*\|\s*Description\s*QTY\s*(.*?)\s*(?:FMP[A-Z0-9]+|Tax Invoice|AWB No\.)", text, re.I | re.S)
        if not match:
            return None, None

        cleaned_lines = []
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r"\d+", stripped):
                continue
            cleaned_lines.append(stripped)

        if not cleaned_lines:
            return None, None

        descriptor = " ".join(cleaned_lines)
        descriptor = re.sub(r"^\d+\s+", "", descriptor)
        parts = [part.strip() for part in descriptor.split("|") if part.strip()]
        if not parts:
            return None, None

        sku_code = parts[0]
        product_name = " | ".join(parts[1:]).strip() if len(parts) > 1 else None
        return sku_code, product_name

    def _extract_sku_from_suborder_line(self, text: str) -> str | None:
        match = re.search(r"SUBORDER CODE\s*QUANTITY\s*([0-9A-Z]+)\s*\|\s*([A-Z0-9\-]+)", text, re.I | re.S)
        if match:
            return match.group(2).strip()
        return None

    def _extract_city_from_address(self, address_block: str | None) -> str | None:
        if not address_block:
            return None
        lines = [line.strip(" ,") for line in address_block.splitlines() if line.strip()]
        for line in reversed(lines):
            city_match = re.search(r"([A-Za-z][A-Za-z\s]+)\s*-\s*\d{6}", line)
            if city_match:
                return city_match.group(1).strip()
        return None

    def _split_city_state(self, city_state: str | None) -> tuple[str | None, str | None]:
        if not city_state:
            return None, None
        if "," in city_state:
            city, state = city_state.split(",", 1)
            return city.strip(), state.strip()
        return city_state.strip(), None

    def _normalize_date(self, raw_value: str | None) -> str | None:
        if not raw_value:
            return None
        cleaned = " ".join(raw_value.replace("\n", " ").split())
        for fmt in (
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d-%m-%Y, %I:%M %p",
            "%d-%m-%Y, %I:%M",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%d/%m/%Y",
        ):
            try:
                return datetime.strptime(cleaned, fmt).date().isoformat()
            except ValueError:
                continue
        match = re.search(r"(\d{2})-(\d{2})-(\d{4})", cleaned)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month}-{day}"
        month_match = re.search(r"(\d{2})-([A-Z]{3})-(\d{4})", cleaned.upper())
        if month_match:
            try:
                return datetime.strptime(month_match.group(0), "%d-%b-%Y").date().isoformat()
            except ValueError:
                return None
        return None

    def _first_capture(self, text: str, pattern: str, flags: int = re.I) -> str | None:
        match = re.search(pattern, text, flags)
        if not match:
            return None
        return " ".join(match.group(1).split())

    def _to_int(self, value: str | None) -> int | None:
        if not value:
            return None
        digits = re.sub(r"[^\d]", "", value)
        return int(digits) if digits else None

    def _to_float(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return round(float(value.replace(",", "")), 2)
        except ValueError:
            return None

    def _extract_labels(
        self,
        page_image: Image.Image,
        layout: int,
        detection_mode: str,
        text_label_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[Image.Image]:
        if text_label_boxes:
            return [page_image.crop(box) for box in text_label_boxes]

        detected_labels = self._detect_shipping_labels(page_image)
        if detected_labels:
            return detected_labels

        courier_labels = self._extract_courier_labels(page_image, layout)
        if courier_labels:
            return courier_labels

        if detection_mode == "smart":
            smart_labels = self._smart_detect_labels(page_image)
            if smart_labels:
                return smart_labels
        return self._basic_split_labels(page_image, layout)

    def _find_text_label_boxes(self, page: fitz.Page, scale: float) -> list[tuple[int, int, int, int]]:
        page_width = page.rect.width
        left_blocks = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            normalized_text = " ".join(text.split()).upper()
            if not normalized_text:
                continue
            if x1 > page_width * 0.52:
                continue
            if "TAX INVOICE" in normalized_text:
                continue
            left_blocks.append((x0, y0, x1, y1, normalized_text))

        if not left_blocks:
            return []

        left_text = " ".join(block[4] for block in left_blocks)
        anchor_terms = ["DELIVERY ADDRESS", "EKART", "FLIPKART", "SELLER GSTIN", "COD"]
        if not any(term in left_text for term in anchor_terms):
            return []

        x0 = min(block[0] for block in left_blocks)
        y0 = min(block[1] for block in left_blocks)
        x1 = max(block[2] for block in left_blocks)
        y1 = max(block[3] for block in left_blocks)
        pad_x = max(4.0, (x1 - x0) * 0.04)
        pad_y = max(4.0, (y1 - y0) * 0.04)

        return [
            (
                max(0, int((x0 - pad_x) * scale)),
                max(0, int((y0 - pad_y) * scale)),
                int(min(page.rect.width * scale, (x1 + pad_x) * scale)),
                int(min(page.rect.height * scale, (y1 + pad_y) * scale)),
            )
        ]

    def _find_vector_label_boxes(self, page: fitz.Page, scale: float) -> list[tuple[int, int, int, int]]:
        page_text = page.get_text().upper()
        if "ORDERED THROUGH" in page_text or "FLIPKART" in page_text or "SHOPSY" in page_text:
            return self._find_flipkart_vector_box(page, scale)
        if "TAX INVOICE (ORIGINAL FOR RECIPIENT)" in page_text and "EKART" in page_text:
            return self._find_snapdeal_vector_box(page, scale)

        return self._find_anchor_vector_box(page, scale)

    def _find_snapdeal_vector_box(self, page: fitz.Page, scale: float) -> list[tuple[int, int, int, int]]:
        candidate_rects = []
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            fill = drawing.get("fill")
            if not rect or not fill:
                continue
            if max(fill) > 0.1:
                continue
            if rect.x1 > page.rect.width * 0.64:
                continue
            if rect.width > page.rect.width * 0.75 or rect.height > page.rect.height * 0.85:
                continue
            candidate_rects.append(rect)

        if not candidate_rects:
            return []

        x0 = min(rect.x0 for rect in candidate_rects)
        y0 = min(rect.y0 for rect in candidate_rects)
        x1 = max(rect.x1 for rect in candidate_rects)
        y1 = max(rect.y1 for rect in candidate_rects)
        if (x1 - x0) < page.rect.width * 0.38 or (y1 - y0) < page.rect.height * 0.45:
            return []
        return [self._scaled_box(page, scale, x0, y0, x1, y1, 0.01, 0.01)]

    def _find_flipkart_vector_box(self, page: fitz.Page, scale: float) -> list[tuple[int, int, int, int]]:
        return self._find_anchor_vector_box(page, scale)

    def _find_anchor_vector_box(self, page: fitz.Page, scale: float) -> list[tuple[int, int, int, int]]:
        anchor_terms = [
            "DELIVERY ADDRESS",
            "EKART",
            "SELLER GSTIN",
            "ORDERED THROUGH",
            "FLIPKART",
            "SHOPSY",
            "E-KART LOGISTICS",
            "COD",
        ]
        anchor_rects = []
        for term in anchor_terms:
            anchor_rects.extend(page.search_for(term))

        if not anchor_rects:
            return []

        ax0 = min(rect.x0 for rect in anchor_rects)
        ay0 = min(rect.y0 for rect in anchor_rects)
        ax1 = max(rect.x1 for rect in anchor_rects)
        ay1 = max(rect.y1 for rect in anchor_rects)
        expand_x = max(12.0, (ax1 - ax0) * 0.18)
        expand_y = max(12.0, (ay1 - ay0) * 0.18)
        search_rect = fitz.Rect(ax0 - expand_x, ay0 - expand_y, ax1 + expand_x, ay1 + expand_y)

        candidate_rects = []
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            fill = drawing.get("fill")
            if not rect:
                continue
            if not rect.intersects(search_rect):
                continue
            if fill is not None and max(fill) > 0.4:
                continue
            if rect.width > page.rect.width * 0.85 or rect.height > page.rect.height * 0.9:
                continue
            candidate_rects.append(rect)

        if not candidate_rects:
            return []

        x0 = min(rect.x0 for rect in candidate_rects)
        y0 = min(rect.y0 for rect in candidate_rects)
        x1 = max(rect.x1 for rect in candidate_rects)
        y1 = max(rect.y1 for rect in candidate_rects)

        width = x1 - x0
        height = y1 - y0
        if width < page.rect.width * 0.22 or height < page.rect.height * 0.28:
            return []

        return [self._scaled_box(page, scale, x0, y0, x1, y1, 0.01, 0.01)]

    def _scaled_box(
        self,
        page: fitz.Page,
        scale: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        pad_x_ratio: float,
        pad_y_ratio: float,
    ) -> tuple[int, int, int, int]:
        width = x1 - x0
        height = y1 - y0
        pad_x = max(3.0, width * pad_x_ratio)
        pad_y = max(3.0, height * pad_y_ratio)
        return (
            max(0, int((x0 - pad_x) * scale)),
            max(0, int((y0 - pad_y) * scale)),
            int(min(page.rect.width * scale, (x1 + pad_x) * scale)),
            int(min(page.rect.height * scale, (y1 + pad_y) * scale)),
        )

    def _detect_shipping_labels(self, page_image: Image.Image) -> list[Image.Image]:
        rgb = np.array(page_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bordered_labels = self._detect_bordered_labels(page_image, gray)
        if bordered_labels:
            return bordered_labels

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        page_area = page_image.width * page_image.height
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < page_area * 0.08:
                continue
            if w < page_image.width * 0.18 or h < page_image.height * 0.25:
                continue
            ratio = w / max(h, 1)
            if ratio < 0.45 or ratio > 1.4:
                continue

            roi_thresh = thresh[y : y + h, x : x + w]
            roi_gray = gray[y : y + h, x : x + w]
            border_score = self._border_strength(roi_thresh)
            barcode_score = self._barcode_signature(roi_gray)
            fill_ratio = cv2.countNonZero(roi_thresh) / max(area, 1)
            left_bias = 1.15 - (x / max(page_image.width, 1))
            score = (area / page_area) * 2.5 + border_score * 2.4 + barcode_score * 2.8 + fill_ratio + left_bias
            candidates.append((score, (x, y, w, h)))

        if not candidates:
            return []

        boxes = [box for _, box in sorted(candidates, key=lambda item: item[0], reverse=True)]
        boxes = self._dedupe_boxes(boxes)
        scored_boxes = []
        for box in boxes:
            for score, candidate_box in candidates:
                if box == candidate_box:
                    scored_boxes.append((score, box))
                    break
        scored_boxes.sort(key=lambda item: item[0], reverse=True)

        extracted: list[Image.Image] = []
        for _, (x, y, w, h) in scored_boxes[:4]:
            pad_x = max(8, int(w * 0.02))
            pad_y = max(8, int(h * 0.02))
            crop = page_image.crop(
                (
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(page_image.width, x + w + pad_x),
                    min(page_image.height, y + h + pad_y),
                )
            )
            extracted.append(crop)
        return extracted

    def _detect_bordered_labels(self, page_image: Image.Image, gray: np.ndarray) -> list[Image.Image]:
        edges = cv2.Canny(gray, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        page_area = page_image.width * page_image.height
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area < page_area * 0.05:
                continue
            if w < page_image.width * 0.12 or h < page_image.height * 0.18:
                continue

            ratio = w / max(h, 1)
            if ratio < 0.3 or ratio > 1.5:
                continue

            rectangularity = cv2.contourArea(contour) / max(area, 1)
            if rectangularity < 0.65:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            barcode_score = self._barcode_signature(roi_gray)
            border_score = rectangularity
            left_bias = 1.1 - (x / max(page_image.width, 1))
            score = (area / page_area) * 3.5 + border_score * 2.2 + barcode_score * 2.8 + left_bias
            candidates.append((score, (x, y, w, h)))

        if not candidates:
            return []

        boxes = [box for _, box in sorted(candidates, key=lambda item: item[0], reverse=True)]
        boxes = self._dedupe_boxes(boxes)
        extracted: list[Image.Image] = []
        for x, y, w, h in boxes[:6]:
            pad_x = max(6, int(w * 0.015))
            pad_y = max(6, int(h * 0.015))
            crop = page_image.crop(
                (
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(page_image.width, x + w + pad_x),
                    min(page_image.height, y + h + pad_y),
                )
            )
            extracted.append(crop)
        return extracted

    def _border_strength(self, roi_thresh: np.ndarray) -> float:
        h, w = roi_thresh.shape
        band = max(2, min(h, w) // 40)
        mask = np.zeros_like(roi_thresh)
        mask[:band, :] = 255
        mask[-band:, :] = 255
        mask[:, :band] = 255
        mask[:, -band:] = 255
        border_pixels = cv2.countNonZero(cv2.bitwise_and(roi_thresh, mask))
        border_area = cv2.countNonZero(mask)
        return border_pixels / max(border_area, 1)

    def _barcode_signature(self, roi_gray: np.ndarray) -> float:
        h, w = roi_gray.shape
        top_region = roi_gray[int(h * 0.1) : int(h * 0.34), int(w * 0.05) : int(w * 0.86)]
        lower_region = roi_gray[int(h * 0.58) : int(h * 0.88), int(w * 0.38) : int(w * 0.9)]
        regions = [region for region in (top_region, lower_region) if region.size]
        if not regions:
            return 0.0

        scores: list[float] = []
        for region in regions:
            sobel = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
            scores.append(float(np.mean(np.abs(sobel)) / 255.0))
        return sum(scores) / len(scores)

    def _extract_courier_labels(self, page_image: Image.Image, layout: int) -> list[Image.Image]:
        rgb = np.array(page_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = page_image.width * page_image.height
        blocks: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < page_area * 0.08:
                continue
            if w < page_image.width * 0.35:
                continue
            if h < page_image.height * 0.12:
                continue
            blocks.append((x, y, w, h))

        if not blocks:
            return []

        blocks = self._dedupe_boxes(blocks)
        blocks.sort(key=lambda box: box[1])
        extracted: list[Image.Image] = []
        for x, y, w, h in blocks:
            pad_x = max(8, int(w * 0.015))
            pad_y = max(8, int(h * 0.02))
            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(page_image.width, x + w + pad_x)
            bottom = min(page_image.height, y + h + pad_y)
            block = page_image.crop((left, top, right, bottom))
            extracted.append(self._remove_invoice_panel(block, layout))
        return extracted

    def _basic_split_labels(self, page_image: Image.Image, layout: int) -> list[Image.Image]:
        rows, cols = self._grid_shape(layout)
        width, height = page_image.size
        labels: list[Image.Image] = []

        cell_width = width / cols
        cell_height = height / rows
        for row in range(rows):
            for col in range(cols):
                left = int(round(col * cell_width))
                upper = int(round(row * cell_height))
                right = int(round((col + 1) * cell_width))
                lower = int(round((row + 1) * cell_height))
                labels.append(page_image.crop((left, upper, right, lower)))
        return labels

    def _smart_detect_labels(self, page_image: Image.Image) -> list[Image.Image]:
        working_image = page_image
        scale_back = 1.0
        max_dimension = max(page_image.size)
        if max_dimension > SMART_DETECTION_MAX_DIMENSION:
            scale_back = max_dimension / SMART_DETECTION_MAX_DIMENSION
            resized_size = (
                max(1, int(page_image.width / scale_back)),
                max(1, int(page_image.height / scale_back)),
            )
            working_image = page_image.resize(resized_size, Image.Resampling.BILINEAR)

        rgb = np.array(working_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = working_image.size[0] * working_image.size[1]
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < page_area * 0.04:
                continue
            if w < 150 or h < 150:
                continue
            ratio = w / max(h, 1)
            if ratio < 0.35 or ratio > 3.5:
                continue
            boxes.append((x, y, w, h))

        # Keep the largest distinct rectangles and sort them in reading order.
        boxes = self._dedupe_boxes(boxes)
        boxes.sort(key=lambda box: (round(box[1] / 100), box[0]))
        scaled_boxes = [
            (
                int(round(x * scale_back)),
                int(round(y * scale_back)),
                int(round(w * scale_back)),
                int(round(h * scale_back)),
            )
            for x, y, w, h in boxes
        ]
        return [page_image.crop((x, y, x + w, y + h)) for x, y, w, h in scaled_boxes]

    def _dedupe_boxes(self, boxes: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        deduped: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            if any(self._iou(box, existing) > 0.65 for existing in deduped):
                continue
            deduped.append(box)
        return deduped

    def _iou(self, first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        fx, fy, fw, fh = first
        sx, sy, sw, sh = second
        left = max(fx, sx)
        top = max(fy, sy)
        right = min(fx + fw, sx + sw)
        bottom = min(fy + fh, sy + sh)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        union = fw * fh + sw * sh - intersection
        return intersection / union if union else 0.0

    def _normalize_label(self, label: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(label)
        bbox = ImageOps.invert(grayscale.point(lambda value: 255 if value < 245 else 0)).getbbox()
        if bbox:
            label = label.crop(bbox)
        return ImageOps.expand(label.convert("RGB"), border=8, fill="white")

    def _find_invoice_free_area(
        self,
        merged: np.ndarray,
        width: int,
        height: int,
        layout: int,
    ) -> tuple[int, int, int, int] | None:
        column_density = np.count_nonzero(merged, axis=0) / max(height, 1)
        active_columns = column_density > 0.03

        spans: list[tuple[int, int]] = []
        start: int | None = None
        min_span_width = max(24, int(width * 0.08))
        for index, is_active in enumerate(active_columns):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                if index - start >= min_span_width:
                    spans.append((start, index))
                start = None
        if start is not None and width - start >= min_span_width:
            spans.append((start, width))

        if len(spans) < 2:
            return None

        left_span = next((span for span in spans if span[0] <= width * 0.12), None)
        if left_span is None:
            return None

        right_candidates = [span for span in spans if span[0] > left_span[1]]
        if not right_candidates:
            return None

        right_span = max(right_candidates, key=lambda span: span[1] - span[0])
        left_width = left_span[1] - left_span[0]
        right_width = right_span[1] - right_span[0]
        gap_width = right_span[0] - left_span[1]
        max_right_width_ratio = 0.72
        if left_width < width * 0.2:
            return None
        if left_width >= right_width:
            return None
        if right_width > width * max_right_width_ratio:
            return None
        if gap_width < width * 0.025:
            return None

        left_roi = merged[:, left_span[0] : left_span[1]]
        right_roi = merged[:, right_span[0] : right_span[1]]
        left_fill = cv2.countNonZero(left_roi) / max(left_roi.size, 1)
        right_fill = cv2.countNonZero(right_roi) / max(right_roi.size, 1)
        if right_fill >= left_fill * 1.35:
            return None

        row_density = np.count_nonzero(left_roi, axis=1) / max(left_width, 1)
        active_rows = row_density > 0.02
        top = next((index for index, active in enumerate(active_rows) if active), 0)
        bottom = height - next((index for index, active in enumerate(reversed(active_rows)) if active), 0)
        if bottom - top < height * 0.35:
            return None

        pad_x = max(8, int(left_width * 0.025))
        pad_y = max(8, int((bottom - top) * 0.03))
        left = max(0, left_span[0] - pad_x)
        right = min(width, left_span[1] + pad_x)
        top = max(0, top - pad_y)
        bottom = min(height, bottom + pad_y)
        return left, top, right - left, bottom - top

    def _extract_primary_label_area(self, label: Image.Image) -> Image.Image:
        if label.width < label.height * 1.1:
            return label

        rgb = np.array(label.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = label.width * label.height
        content_box = self._find_main_content_columns(merged, label.width, label.height)
        if content_box is not None:
            x, y, w, h = content_box
            if w < label.width * 0.78:
                return label
            return label.crop((x, y, x + w, y + h))

        best_box: tuple[int, int, int, int] | None = None
        best_score = 0.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < image_area * 0.15:
                continue
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.55 or aspect_ratio > 1.8:
                continue

            roi = merged[y : y + h, x : x + w]
            fill_ratio = cv2.countNonZero(roi) / max(area, 1)
            # Favor large rectangular label cards rather than sparse side text.
            score = area * max(fill_ratio, 0.2)
            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        if not best_box:
            return label

        x, y, w, h = best_box
        pad_x = max(6, int(w * 0.02))
        pad_y = max(6, int(h * 0.02))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(label.width, x + w + pad_x)
        bottom = min(label.height, y + h + pad_y)
        return label.crop((left, top, right, bottom))

    def _find_main_content_columns(
        self,
        merged: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        column_density = np.count_nonzero(merged, axis=0) / max(height, 1)
        active_columns = column_density > 0.05

        spans: list[tuple[int, int]] = []
        start: int | None = None
        for index, is_active in enumerate(active_columns):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                if index - start > max(20, int(width * 0.08)):
                    spans.append((start, index))
                start = None
        if start is not None and width - start > max(20, int(width * 0.08)):
            spans.append((start, width))

        if not spans:
            return None

        scored_spans: list[tuple[float, tuple[int, int]]] = []
        for left, right in spans:
            span_width = right - left
            roi = merged[:, left:right]
            fill_ratio = cv2.countNonZero(roi) / max(roi.size, 1)
            left_bias = 1.25 - (left / max(width, 1))
            score = (span_width * fill_ratio) * left_bias
            scored_spans.append((score, (left, right)))

        _, (best_left, best_right) = max(scored_spans, key=lambda item: item[0])
        roi = merged[:, best_left:best_right]
        row_density = np.count_nonzero(roi, axis=1) / max(best_right - best_left, 1)
        active_rows = row_density > 0.03
        top = next((index for index, active in enumerate(active_rows) if active), 0)
        bottom = height - next((index for index, active in enumerate(reversed(active_rows)) if active), 0)

        pad_x = max(6, int((best_right - best_left) * 0.02))
        pad_y = max(6, int((bottom - top) * 0.02))
        left = max(0, best_left - pad_x)
        right = min(width, best_right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(height, bottom + pad_y)
        if right - left < width * 0.25 or bottom - top < height * 0.25:
            return None
        return left, top, right - left, bottom - top

    def _create_output_pdf(
        self,
        labels: list[Image.Image],
        output_pdf_path: Path,
        layout: int,
        paper_size: str,
    ) -> tuple[int, list[Image.Image]]:
        page_size = PAGE_SIZES[paper_size]
        page_width, page_height = page_size
        margin = 24 if layout != 2 else 10
        gap = 12 if layout != 2 else 8
        rows, cols = self._grid_shape(layout)
        usable_width = page_width - (margin * 2) - (gap * (cols - 1))
        usable_height = page_height - (margin * 2) - (gap * (rows - 1))
        cell_width = usable_width / cols
        cell_height = usable_height / rows

        pdf = canvas.Canvas(str(output_pdf_path), pagesize=page_size)
        slots_per_page = rows * cols
        output_page_count = max(1, math.ceil(len(labels) / slots_per_page)) if labels else 1
        preview_images: list[Image.Image] = []

        for page_start in range(0, max(len(labels), 1), slots_per_page):
            batch = labels[page_start : page_start + slots_per_page]
            if not batch and page_start > 0:
                break
            if not batch:
                batch = []

            preview_page = Image.new("RGB", (PREVIEW_IMAGE_WIDTH, int(PREVIEW_IMAGE_WIDTH * (page_height / page_width))), "white")
            preview_draw_scale = PREVIEW_IMAGE_WIDTH / page_width

            for slot_index, label in enumerate(batch):
                row = slot_index // cols
                col = slot_index % cols
                x = margin + (col * (cell_width + gap))
                y = page_height - margin - cell_height - (row * (cell_height + gap))
                # Resize into a high-resolution raster so the final PDF stays sharp.
                prepared = self._fit_label_to_cell(label, cell_width, cell_height, allow_rotate=layout != 2)
                px_per_point = OUTPUT_TARGET_DPI / 72
                draw_width = prepared.width / px_per_point
                draw_height = prepared.height / px_per_point
                if layout == 2:
                    x_offset = x
                    y_offset = y + (cell_height - draw_height)
                else:
                    x_offset = x + (cell_width - draw_width) / 2
                    y_offset = y + (cell_height - draw_height) / 2
                pdf.drawImage(
                    ImageReader(prepared),
                    x_offset,
                    y_offset,
                    width=draw_width,
                    height=draw_height,
                    preserveAspectRatio=True,
                    anchor="c",
                )
                preview_label = prepared.copy()
                preview_label.thumbnail(
                    (
                        max(1, int(draw_width * preview_draw_scale)),
                        max(1, int(draw_height * preview_draw_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                preview_x = int(x_offset * preview_draw_scale)
                preview_y = int((page_height - y_offset - draw_height) * preview_draw_scale)
                preview_page.paste(preview_label, (preview_x, preview_y))

            pdf.showPage()
            preview_images.append(preview_page)

        pdf.save()
        return output_page_count, preview_images

    def _fit_label_to_cell(
        self,
        label: Image.Image,
        cell_width: float,
        cell_height: float,
        allow_rotate: bool = True,
    ) -> Image.Image:
        px_per_point = OUTPUT_TARGET_DPI / 72
        target_width = max(1, int(cell_width * px_per_point))
        target_height = max(1, int(cell_height * px_per_point))

        tightened = ImageOps.crop(label, border=6) if min(label.size) > 24 else label
        candidates = [tightened]
        if allow_rotate and ((label.width > label.height and target_width < target_height) or (
            label.height > label.width and target_height < target_width
        )):
            candidates.append(tightened.rotate(90, expand=True))

        best = max(
            candidates,
            key=lambda image: min(target_width / image.width, target_height / image.height),
        )
        fitted = best.copy()
        fitted.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        return fitted

    def _grid_shape(self, layout: int) -> tuple[int, int]:
        if layout == 2:
            return 2, 1
        if layout == 6:
            return 3, 2
        return 2, 2

    def _clean_filename(self, filename: str) -> str:
        safe = "".join(char for char in filename if char.isalnum() or char in {".", "-", "_"})
        return safe or "upload.pdf"
