from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def create_demo_pdf(output_path: str = "sample-labels.pdf") -> None:
    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    labels = [
        (15 * mm, 160 * mm, 85 * mm, 55 * mm, "Order #1001"),
        (110 * mm, 160 * mm, 85 * mm, 55 * mm, "Order #1002"),
        (15 * mm, 90 * mm, 85 * mm, 55 * mm, "Order #1003"),
        (110 * mm, 90 * mm, 85 * mm, 55 * mm, "Order #1004"),
    ]

    for x, y, w, h, title in labels:
        pdf.roundRect(x, y, w, h, 6 * mm, stroke=1, fill=0)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(x + 8 * mm, y + h - 12 * mm, title)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(x + 8 * mm, y + h - 22 * mm, "Recipient: Demo Customer")
        pdf.drawString(x + 8 * mm, y + h - 30 * mm, "Address: 42 Example Street")
        pdf.drawString(x + 8 * mm, y + h - 38 * mm, "City: Mumbai 400001")
        pdf.rect(x + 8 * mm, y + 8 * mm, w - 16 * mm, 12 * mm, stroke=1, fill=0)
        pdf.drawCentredString(x + (w / 2), y + 12 * mm, "|||| || ||| |||| |||")

    pdf.showPage()
    pdf.save()
    print(f"Created sample PDF at {output_path} ({width} x {height} points)")


if __name__ == "__main__":
    create_demo_pdf()
