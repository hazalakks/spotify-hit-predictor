"""Convert PROJECT_REPORT.md to an optimized, compact PDF with embedded figures."""
import os
import fitz
from markdown_pdf import MarkdownPdf, Section

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(ROOT, "PROJECT_REPORT.md"), encoding="utf-8").read()

css = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 16pt; color: #1DB954; }
h2 { font-size: 12.5pt; border-bottom: 1px solid #ccc; margin-top: 10px; }
table { border-collapse: collapse; font-size: 8.5pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 5px; }
th { background: #eef6f0; }
code { font-size: 8.5pt; }
img { display: block; margin: 4px auto; }
"""

temp_pdf = os.path.join(ROOT, "PROJECT_REPORT_raw.pdf")
out_pdf = os.path.join(ROOT, "PROJECT_REPORT.pdf")

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(Section(md, root=ROOT, paper_size="A4"), user_css=css)
pdf.meta["title"] = "Spotify Hit Tahmini - Proje Raporu"
pdf.meta["author"] = "Hazal"
pdf.save(temp_pdf)

# Compress and deflate streams with PyMuPDF
doc = fitz.open(temp_pdf)
doc.save(out_pdf, garbage=4, deflate=True, clean=True)
doc.close()

if os.path.exists(temp_pdf):
    os.remove(temp_pdf)

size_mb = os.path.getsize(out_pdf) / (1024 * 1024)
print(f"Saved optimized PDF: {out_pdf} ({size_mb:.2f} MB)")
