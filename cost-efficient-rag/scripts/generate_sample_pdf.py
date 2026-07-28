"""
Generate ``data/raw_documents/onboarding_guide.pdf`` — a small, real, valid
PDF used as the sample corpus's PDF-format document.

Why this exists: the assignment requires demonstrating PDF ingestion, but
authoring a PDF by hand (rather than exporting one from Word/Google Docs)
needs *some* tool to produce valid PDF bytes. Rather than adding a new
dependency (e.g. reportlab) purely to generate one sample fixture,
this uses low-level object construction already available in ``pypdf``
(a dependency the project needs anyway, for *reading* PDFs) to build a
minimal content stream by hand. This is a one-time data-prep script, not
part of the application itself — the app's ``ingestion.load_pdf`` only
ever reads PDFs, never writes them.

Run: ``python scripts/generate_sample_pdf.py``
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

PAGES_TEXT: list[str] = [
    "Synergy Labs Onboarding Guide\n"
    "\n"
    "Step 1: Create Your Account\n"
    "Go to app.synergylabs.io/signup and enter your work email address.\n"
    "You must verify your email within 24 hours or the signup link expires.\n"
    "\n"
    "Step 2: Connect a Data Source\n"
    "From the Integrations tab, choose a connector such as PostgreSQL, Snowflake, "
    "or a REST API. The first sync typically completes within 10 minutes for "
    "datasets under 1 GB.",
    "Step 3: Invite Your Team\n"
    "Navigate to Settings > Members and enter teammate email addresses. Each "
    "invited member consumes one seat on your plan. Invitations expire after 7 days "
    "if not accepted.\n"
    "\n"
    "Step 4: Configure Alerts\n"
    "Under Monitoring > Alerts, define a threshold rule (for example, error rate "
    "above 5 percent) and choose a delivery channel: email, Slack, or PagerDuty. "
    "Alerts are evaluated every 60 seconds.\n"
    "\n"
    "Step 5: Go Live\n"
    "Once your data source, team, and alerts are configured, click Activate "
    "Workspace on the dashboard. Activation cannot be undone without contacting "
    "support.",
]


def _build_content_stream(text: str) -> bytes:
    """Very small hand-rolled PDF content stream: positions text lines with Td/T*."""
    lines = text.split("\n")
    stream_parts = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
    for i, line in enumerate(lines):
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i > 0:
            stream_parts.append("T*")
        stream_parts.append(f"({escaped}) Tj")
    stream_parts.append("ET")
    return "\n".join(stream_parts).encode("latin-1", errors="replace")


def generate_pdf(pages_text: list[str], output_path: str | Path) -> None:
    """Write a multi-page PDF where each page renders one plain-text block."""
    writer = PdfWriter()

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)

        stream_obj = DecodedStreamObject()
        stream_obj.set_data(_build_content_stream(text))
        stream_ref = writer._add_object(stream_obj)

        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font_dict)

        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts

        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream_ref

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "raw_documents" / "onboarding_guide.pdf"
    generate_pdf(PAGES_TEXT, target)
    print(f"Wrote {target}")
