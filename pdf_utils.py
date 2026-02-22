
# pdf_utils.py
# pdf_utils.py  —  Professional PDF Report Generator
# Proper structured layout for both English and Telugu reports.
# Telugu requires NotoSans Telugu font — auto-downloaded on first use in Colab.

import os
import re
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import cm
from reportlab.lib           import colors
from reportlab.pdfbase       import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums     import TA_CENTER, TA_LEFT

# ── Telugu font setup ─────────────────────────────────────────────────────────
_TELUGU_FONT_REGISTERED = False
_FONT_NAME_TE  = "NotoSansTelugu"
_FONT_URL      = (
    "https://github.com/googlefonts/noto-fonts/raw/main/"
    "hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf"
)
_FONT_SEARCH_PATHS = [
    "/content/NotoSansTelugu-Regular.ttf",
    "/content/Enhanced_XRay_Project/NotoSansTelugu-Regular.ttf",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "NotoSansTelugu-Regular.ttf"),
    os.path.expanduser("~/NotoSansTelugu-Regular.ttf"),
]

def _ensure_telugu_font():
    global _TELUGU_FONT_REGISTERED
    if _TELUGU_FONT_REGISTERED:
        return True
    for path in _FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(_FONT_NAME_TE, path))
                _TELUGU_FONT_REGISTERED = True
                print(f"✅ Telugu font loaded: {path}")
                return True
            except Exception as e:
                print(f"   ⚠ Font load failed ({path}): {e}")
    try:
        import urllib.request
        save_to = _FONT_SEARCH_PATHS[1]
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        print("⬇ Downloading NotoSans Telugu font ...")
        urllib.request.urlretrieve(_FONT_URL, save_to)
        pdfmetrics.registerFont(TTFont(_FONT_NAME_TE, save_to))
        _TELUGU_FONT_REGISTERED = True
        print("✅ Telugu font downloaded and registered")
        return True
    except Exception as e:
        print(f"   ⚠ Telugu font not available: {e}")
        return False


# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK      = colors.HexColor("#1e3a5f")
C_ACCENT    = colors.HexColor("#4f46e5")
C_LABEL     = colors.HexColor("#374151")
C_BODY      = colors.HexColor("#111827")
C_MUTED     = colors.HexColor("#6b7280")
C_GREEN     = colors.HexColor("#065f46")
C_HEADER_BG = colors.HexColor("#eef2ff")
C_ROW_ALT   = colors.HexColor("#f9fafb")


# ── Style factory ──────────────────────────────────────────────────────────────
def _make_styles(font_name, is_telugu):
    bold = font_name if is_telugu else font_name + "-Bold"
    lead = 17 if is_telugu else 15

    return {
        "title": ParagraphStyle("St", fontName=bold, fontSize=17, leading=22,
            alignment=TA_CENTER, textColor=C_DARK, spaceAfter=0),
        "subtitle": ParagraphStyle("Ss", fontName=font_name, fontSize=10, leading=14,
            alignment=TA_CENTER, textColor=C_ACCENT, spaceAfter=2),
        "section": ParagraphStyle("Sh", fontName=bold, fontSize=11, leading=15,
            textColor=C_DARK, spaceBefore=0, spaceAfter=0),
        "label": ParagraphStyle("Sl", fontName=bold, fontSize=10, leading=lead,
            textColor=C_LABEL),
        "value": ParagraphStyle("Sv", fontName=font_name, fontSize=10, leading=lead,
            textColor=C_BODY),
        "body": ParagraphStyle("Sb", fontName=font_name, fontSize=10, leading=lead,
            textColor=C_BODY, spaceAfter=3),
        "findkey": ParagraphStyle("Sfk", fontName=bold, fontSize=10, leading=lead,
            textColor=C_ACCENT, spaceBefore=5, spaceAfter=1),
        "numbered": ParagraphStyle("Sn", fontName=font_name, fontSize=10, leading=lead,
            textColor=C_BODY, leftIndent=12, spaceAfter=3),
        "disclaimer": ParagraphStyle("Sd", fontName=font_name, fontSize=8.5, leading=13,
            textColor=C_MUTED, alignment=TA_CENTER, spaceBefore=4),
        "confidence": ParagraphStyle("Sc", fontName=bold, fontSize=10, leading=lead,
            textColor=C_GREEN),
    }


# ── Helper flowables ──────────────────────────────────────────────────────────
def _thick_hr(col=None):
    return HRFlowable(width="100%", thickness=2,
                      color=col or C_ACCENT,
                      spaceAfter=5, spaceBefore=5)

def _thin_hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_MUTED,
                      spaceAfter=3, spaceBefore=3)

def _kv(label_txt, value_txt, st):
    tbl = Table(
        [[Paragraph(label_txt, st["label"]),
          Paragraph(str(value_txt), st["value"])]],
        colWidths=[5.5*cm, 10.5*cm],
    )
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ]))
    return tbl

def _section_header(text, st):
    tbl = Table([[Paragraph(text, st["section"])]], colWidths=[16*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_HEADER_BG),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LINEBELOW",    (0,0), (-1,-1), 1.5, C_ACCENT),
    ]))
    return tbl


# ── Public API ─────────────────────────────────────────────────────────────────
def create_pdf(report_text, file_path, language="english"):
    """
    Create a well-formatted A4 PDF from structured report text.
    All =====, ----- separators are rendered as visual HR lines, not raw text.
    Key:value rows are properly aligned in two-column tables.
    Telugu text uses NotoSansTelugu font (auto-downloaded).
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        is_telugu = (language.lower() == "telugu")

        if is_telugu and _ensure_telugu_font():
            font_name = _FONT_NAME_TE
        else:
            font_name = "Helvetica"

        st    = _make_styles(font_name, is_telugu)
        story = _build_story(report_text, st, is_telugu)

        doc = SimpleDocTemplate(
            file_path,
            pagesize     = A4,
            rightMargin  = 2*cm,  leftMargin  = 2*cm,
            topMargin    = 2*cm,  bottomMargin= 2*cm,
            title        = "AI X-Ray Analysis Report",
        )
        doc.build(story)
        print(f"✅ PDF created: {file_path}")
        return True

    except Exception as e:
        print(f"❌ PDF creation error: {e}")
        import traceback; traceback.print_exc()
        try:
            txt = file_path.replace(".pdf", "_report.txt")
            open(txt, "w", encoding="utf-8").write(report_text)
            print(f"   ↳ Fallback text file: {txt}")
        except Exception:
            pass
        return False


# ── Story builder ─────────────────────────────────────────────────────────────

_SECTION_EN = {
    "PATIENT DETAILS", "YOUR DETAILS", "STUDY INFORMATION",
    "AI CONFIDENCE SCORE",
    "1. INDICATION", "2. TECHNIQUE", "3. COMPARISON",
    "4. FINDINGS", "5. IMPRESSION", "6. RECOMMENDATIONS",
    "WHAT WE FOUND (IN SIMPLE TERMS)", "DETAILED FINDINGS",
    "DOCTOR'S CONCLUSION", "WHAT YOU SHOULD DO",
    "IMPORTANT REMINDER", "DISCLAIMER",
}
_SECTION_TE = {
    "రోగి వివరాలు", "మీ వివరాలు", "పరీక్షా సమాచారం",
    "AI విశ్వాసనీయతా స్కోర్",
    "1. సూచన", "2. పద్ధతి", "3. పోలిక",
    "4. లక్షణాలు", "5. అభిప్రాయం", "6. సిఫార్సులు",
    "మేము ఏమి కనుగొన్నాము (సరళమైన భాషలో)", "వివరణాత్మక లక్షణాలు",
    "వైద్య నిర్ణయం", "మీరు ఏమి చేయాలి",
    "ముఖ్యమైన గుర్తు", "నిరాకరణ",
}
_TITLE_LINES = {
    "RADIOLOGY REPORT", "YOUR X-RAY RESULTS REPORT",
    "AI-Assisted Diagnostic Imaging System",
    "రేడియాలజీ నివేదిక", "మీ ఎక్స్-రే ఫలితాల నివేదిక",
    "AI-సహాయక రోగ నిర్ధారణ చిత్రీకరణ వ్యవస్థ",
}


def _build_story(text, st, is_telugu):
    story  = []
    lines  = text.split("\n")
    skeys  = _SECTION_TE if is_telugu else _SECTION_EN

    for i, raw in enumerate(lines):
        line = raw.strip()

        # 1. Separator lines — NEVER printed as text; render as HR lines
        if re.match(r'^[=]{5,}', line) or re.match(r'^[-]{5,}', line):
            if re.match(r'^[=]{5,}', line):
                story.append(_thick_hr())
            else:
                story.append(_thin_hr())
            continue

        # 2. Empty
        if not line:
            story.append(Spacer(1, 4))
            continue

        # 3. Report title line (centred, large)
        if line in _TITLE_LINES:
            if any(k in line for k in ("REPORT", "నివేదిక")):
                story.append(Paragraph(line, st["title"]))
            else:
                story.append(Paragraph(line, st["subtitle"]))
            continue

        # 4. Section heading → coloured background pill
        if line in skeys:
            story.append(Spacer(1, 6))
            story.append(_section_header(line, st))
            story.append(Spacer(1, 4))
            continue

        # 5. Key : Value rows  →  "  Label Text   : some value here"
        kv = re.match(r'^(\s{1,6})(.+?)\s{2,}:\s+(.+)$', raw)
        if kv:
            lbl = kv.group(2).strip() + " :"
            val = kv.group(3).strip()
            story.append(_kv(lbl, val, st))
            continue

        # 6. Single-colon lines like "  Additional recommendation:\n  text"
        # These are short label lines
        if re.match(r'^\s{2,}[A-Za-zఆ-హ].{2,30}:$', raw):
            story.append(Paragraph(line, st["findkey"]))
            continue

        # 7. Numbered items  "   1. Some text"
        if re.match(r'^\s*\d+\.\s', line):
            story.append(Paragraph(line.lstrip(), st["numbered"]))
            continue

        # 8. Sub-section label ending with ":" (findings anatomy labels)
        if line.endswith(":") and 3 < len(line) < 55:
            story.append(Spacer(1, 3))
            story.append(Paragraph(line, st["findkey"]))
            continue

        # 9. Indented lines (findings body text, recommendations)
        if raw.startswith("  ") or raw.startswith("\t"):
            story.append(Paragraph(line, st["body"]))
            continue

        # 10. Default — regular body paragraph
        story.append(Paragraph(line, st["body"]))

    # Footer
    story.append(Spacer(1, 14))
    story.append(_thick_hr(col=C_MUTED))
    footer = (
        "ఈ నివేదిక AI సహాయంతో రూపొందించబడింది. చికిత్స నిర్ణయాలకు "
        "అర్హత కలిగిన వైద్యుడి సలహా తప్పనిసరి."
        if is_telugu else
        "This report is AI-generated. Consult a qualified healthcare "
        "professional before making any clinical decisions."
    )
    story.append(Paragraph(footer, st["disclaimer"]))
    return story

