"""Generate Voice Agent + Telephony Pricing PDF report."""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import date

OUTPUT = "Voice_Agent_Telephony_Report.pdf"

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE       = colors.HexColor("#1d4ed8")
LIGHT_BLUE = colors.HexColor("#eff6ff")
BORDER_BLUE= colors.HexColor("#bfdbfe")
GREEN      = colors.HexColor("#15803d")
LIGHT_GREEN= colors.HexColor("#f0fdf4")
BORDER_GRN = colors.HexColor("#86efac")
ORANGE     = colors.HexColor("#c2410c")
LIGHT_ORG  = colors.HexColor("#fff7ed")
BORDER_ORG = colors.HexColor("#fed7aa")
YELLOW     = colors.HexColor("#92400e")
LIGHT_YEL  = colors.HexColor("#fffbeb")
RED        = colors.HexColor("#991b1b")
LIGHT_RED  = colors.HexColor("#fef2f2")
GRAY_BG    = colors.HexColor("#f9fafb")
GRAY_DARK  = colors.HexColor("#374151")
GRAY_MID   = colors.HexColor("#6b7280")
GRAY_LIGHT = colors.HexColor("#e5e7eb")
WHITE      = colors.white
BLACK      = colors.HexColor("#111827")
PURPLE     = colors.HexColor("#7e22ce")
LIGHT_PUR  = colors.HexColor("#faf5ff")

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
W, H = A4

def sty(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE_STYLE  = sty("ReportTitle",  fontSize=26, leading=32, textColor=WHITE,
                    fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)
SUB_STYLE    = sty("ReportSub",    fontSize=12, leading=16, textColor=colors.HexColor("#bfdbfe"),
                    fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)
DATE_STYLE   = sty("ReportDate",   fontSize=10, leading=14, textColor=colors.HexColor("#93c5fd"),
                    fontName="Helvetica", alignment=TA_CENTER)
H1           = sty("H1",           fontSize=16, leading=20, textColor=BLUE,
                    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
H2           = sty("H2",           fontSize=13, leading=17, textColor=GRAY_DARK,
                    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
H3           = sty("H3",           fontSize=11, leading=14, textColor=BLUE,
                    fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
BODY         = sty("Body",         fontSize=10, leading=15, textColor=GRAY_DARK,
                    fontName="Helvetica", spaceAfter=4)
BODY_SM      = sty("BodySm",       fontSize=9,  leading=13, textColor=GRAY_MID,
                    fontName="Helvetica", spaceAfter=2)
NOTE         = sty("Note",         fontSize=8.5,leading=12, textColor=GRAY_MID,
                    fontName="Helvetica-Oblique", spaceAfter=4)
TH           = sty("TH",           fontSize=9,  leading=12, textColor=WHITE,
                    fontName="Helvetica-Bold", alignment=TA_CENTER)
TD           = sty("TD",           fontSize=9,  leading=13, textColor=GRAY_DARK,
                    fontName="Helvetica")
TD_C         = sty("TD_C",         fontSize=9,  leading=13, textColor=GRAY_DARK,
                    fontName="Helvetica", alignment=TA_CENTER)
TD_B         = sty("TD_B",         fontSize=9,  leading=13, textColor=BLACK,
                    fontName="Helvetica-Bold")
GREEN_TD     = sty("GreenTD",      fontSize=9,  leading=13, textColor=GREEN,
                    fontName="Helvetica-Bold", alignment=TA_CENTER)
RED_TD       = sty("RedTD",        fontSize=9,  leading=13, textColor=RED,
                    fontName="Helvetica-Bold", alignment=TA_CENTER)
ORANGE_TD    = sty("OrangeTD",     fontSize=9,  leading=13, textColor=ORANGE,
                    fontName="Helvetica-Bold", alignment=TA_CENTER)

def th(txt):  return Paragraph(txt, TH)
def td(txt):  return Paragraph(str(txt), TD)
def tdc(txt): return Paragraph(str(txt), TD_C)
def tdb(txt): return Paragraph(str(txt), TD_B)
def tdg(txt): return Paragraph(str(txt), GREEN_TD)
def tdr(txt): return Paragraph(str(txt), RED_TD)
def tdo(txt): return Paragraph(str(txt), ORANGE_TD)

def divider():
    return HRFlowable(width="100%", thickness=1, color=GRAY_LIGHT, spaceAfter=8, spaceBefore=4)

def section_box(text, bg=LIGHT_BLUE, border=BORDER_BLUE, text_color=BLUE):
    style = sty("BoxStyle", fontSize=11, leading=15, textColor=text_color,
                fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=0)
    t = Table([[Paragraph(text, style)]], colWidths=[W - 40*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("BOX",        (0,0), (-1,-1), 1, border),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    return t

BASE_TS = TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  BLUE),
    ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
    ("GRID",          (0,0), (-1,-1), 0.5, GRAY_LIGHT),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
])

# ── Cover page builder ────────────────────────────────────────────────────────
def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # accent bar
    canvas.setFillColor(colors.HexColor("#1e40af"))
    canvas.rect(0, H*0.38, W, 4, fill=1, stroke=0)
    canvas.restoreState()

def later_page(canvas, doc):
    canvas.saveState()
    # thin top bar
    canvas.setFillColor(BLUE)
    canvas.rect(0, H-8*mm, W, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(15*mm, H-5.5*mm, "Voice Agent & Telephony Report — Forage AI")
    canvas.drawRightString(W-15*mm, H-5.5*mm, f"Page {doc.page}")
    # bottom line
    canvas.setStrokeColor(GRAY_LIGHT)
    canvas.setLineWidth(0.5)
    canvas.line(15*mm, 12*mm, W-15*mm, 12*mm)
    canvas.setFillColor(GRAY_MID)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(W/2, 8*mm, f"Confidential — {date.today().strftime('%B %d, %Y')}")
    canvas.restoreState()

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# ─── COVER ───────────────────────────────────────────────────────────────────
story.append(Spacer(1, 60*mm))
story.append(Paragraph("Voice Agent System", TITLE_STYLE))
story.append(Paragraph("Telephony Price Comparison &amp; Technical Report", SUB_STYLE))
story.append(Spacer(1, 8*mm))
story.append(Paragraph(f"Forage AI  ·  {date.today().strftime('%B %d, %Y')}  ·  Prepared by Salomi Mekala", DATE_STYLE))
story.append(Spacer(1, 10*mm))

cover_info_style = sty("CoverInfo", fontSize=10, leading=16, textColor=colors.HexColor("#93c5fd"),
                        fontName="Helvetica", alignment=TA_CENTER)
story.append(Paragraph("Doctor Branch Discovery · Automated Outbound Calling · Indian Hospitals", cover_info_style))
story.append(PageBreak())

# ─── SECTION 1: WHAT WE BUILT ────────────────────────────────────────────────
story.append(Paragraph("1. What We Built", H1))
story.append(divider())
story.append(section_box("Voice Agent — Automated Hospital Branch Finder"))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "The Voice Agent automatically calls Indian hospitals, holds a natural spoken conversation "
    "with the receptionist, and extracts which <b>branch or city</b> a specific doctor is based at — "
    "then saves the result to the database. No human involvement required.",
    BODY))

story.append(Paragraph("System Context", H2))
story.append(Paragraph(
    "This is part of an 8-agent pipeline. Forage AI's team owns <b>Agent 3 (Email)</b> and "
    "<b>Agent 6 (Voice)</b>. The Voice Agent dials hospitals and extracts doctor branch information.",
    BODY))

story.append(Paragraph("Hardware Constraints", H2))
data = [
    [th("Component"), th("Spec"), th("Impact")],
    [td("CPU"), td("Intel i5-1155G7"), td("All AI models run on CPU — no GPU acceleration")],
    [td("RAM"), td("16 GB"), td("Limits model size — must use small models")],
    [td("OS"), td("Windows 11"), td("No Docker — all native installs")],
    [td("GPU"), td("Intel Iris Xe (integrated)"), td("No CUDA — cannot use GPU for AI")],
    [td("Requirement"), td("Open-source only"), td("No paid AI APIs — Whisper, Piper, Ollama")],
]
t = Table(data, colWidths=[45*mm, 55*mm, 70*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 6))

# ─── SECTION 2: HOW THE CALL WORKS ──────────────────────────────────────────
story.append(Paragraph("2. How a Call Works", H1))
story.append(divider())

story.append(Paragraph("Call Flow — Step by Step", H2))
steps = [
    ("1", "run_twilio.py starts", "Warms up Whisper + Piper models, starts FastAPI server on port 8000"),
    ("2", "Twilio REST API", "Script tells Twilio: call this hospital number (+91), webhook = ngrok URL"),
    ("3", "ngrok tunnel", "Twilio sends audio/events to public ngrok URL → forwarded to localhost:8000"),
    ("4", "Hospital picks up", "Twilio plays trial warning (paid = no warning), hospital presses key"),
    ("5", "POST /answer", "Twilio hits our server → we return TwiML: open WebSocket stream"),
    ("6", "WebSocket opens", "Live two-way audio pipe between Twilio and our server"),
    ("7", "Conversation loop", "Hospital speaks → Whisper → Brain → Piper → plays back to hospital"),
    ("8", "Branch saved", "save_branch() stores result, call ends, recording saved to disk"),
]
data = [[th("Step"), th("Component"), th("What Happens")]]
for s, c, w in steps:
    data.append([tdc(s), tdb(c), td(w)])
t = Table(data, colWidths=[15*mm, 45*mm, 110*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 6))

story.append(Paragraph("Role of ngrok", H2))
story.append(Paragraph(
    "Your laptop sits behind a home WiFi router with a private IP (e.g. 192.168.1.5) that the internet "
    "cannot reach. <b>ngrok</b> creates a secure encrypted tunnel giving your laptop a public URL "
    "(e.g. https://protozoan-undoing-rug.ngrok-free.dev). All Twilio webhook requests go through "
    "this URL → tunnel → your server. If ngrok is not running, Twilio cannot reach your server and "
    "plays 'Application error, goodbye'.",
    BODY))

story.append(Paragraph("Command: ngrok http 8000  (run in a separate terminal before the script)", BODY_SM))
story.append(Spacer(1, 4))

# ─── SECTION 3: COMPONENTS ───────────────────────────────────────────────────
story.append(Paragraph("3. Technical Components", H1))
story.append(divider())

data = [
    [th("Component"), th("Technology"), th("Role"), th("CPU Time")],
    [tdb("Speech-to-Text"), td("faster-whisper\n(tiny + small)"), td("Converts hospital speech to text"), tdo("~350ms")],
    [tdb("Brain / AI"),     td("Heuristic + phi4-mini\nvia Ollama"), td("Decides what to say next"), tdo("2ms–10s")],
    [tdb("Text-to-Speech"), td("Piper TTS ONNX\nen_US-lessac-high"), td("Converts agent reply to audio"), tdo("~350ms (0ms cached)")],
    [tdb("Phone calls"),    td("Twilio Media Streams\n+ WebSocket"), td("Dials hospital, streams audio"), td("N/A")],
    [tdb("Tunnel"),         td("ngrok"), td("Exposes localhost to internet for Twilio"), td("~150ms network")],
    [tdb("Server"),         td("FastAPI + uvicorn"), td("Handles webhooks and WebSocket audio"), td("<5ms")],
    [tdb("Recording"),      td("soundfile + numpy"), td("Saves call audio + transcript + JSON"), td("<1s after call")],
]
t = Table(data, colWidths=[35*mm, 45*mm, 65*mm, 25*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 6))

story.append(Paragraph("Latency Breakdown per Turn", H2))
data = [
    [th("Stage"), th("Time"), th("Can Reduce?"), th("How")],
    [td("Silence detection"), tdo("500ms"), tdg("Yes"), td("Reduce frames (risk: cuts off speech early)")],
    [td("Whisper tiny STT"),  tdo("350ms"), tdo("Partial"), td("GPU would make it 20ms")],
    [td("Brain (heuristic)"), tdg("2ms"),   tdg("N/A"), td("Already instant — pure Python")],
    [td("Brain (LLM)"),       tdr("5–10s"), tdg("Yes"), td("Expand heuristics to avoid LLM calls")],
    [td("Piper TTS"),         tdo("350ms"), tdg("Yes"), td("Pre-cache = 0ms for fixed phrases")],
    [td("Network (Twilio)"),  tdo("150ms"), tdr("No"),  td("Fixed internet latency")],
    [td("TOTAL (heuristic)"), tdo("~1.0s"), tdg("Yes"), td("~0.65s with full cache on GPU")],
]
t = Table(data, colWidths=[45*mm, 22*mm, 28*mm, 75*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Note: Twilio trial vs paid has ZERO effect on latency. All lag comes from CPU processing. "
    "A cloud GPU (e.g. Runpod RTX 3080 at $0.20/hr) would reduce total latency to ~300ms.",
    NOTE))

story.append(PageBreak())

# ─── SECTION 4: TELEPHONY PROVIDERS ─────────────────────────────────────────
story.append(Paragraph("4. Telephony Providers Evaluated", H1))
story.append(divider())

story.append(section_box("US / Global Providers — No Monthly Plan, Pay-As-You-Go", LIGHT_BLUE, BORDER_BLUE, BLUE))
story.append(Spacer(1, 6))

data = [
    [th("Provider"), th("Number/mo"), th("Outbound\nIndia/min"), th("Free Trial"), th("KYC?"), th("WebSocket?"), th("Status")],
    [tdb("Twilio 🇺🇸"),    tdc("$1.15"), tdo("$0.022"), tdg("$14.35"), tdg("No"), tdg("Yes"), tdo("Current ⚡")],
    [tdb("Telnyx 🇺🇸"),    tdc("$1.00"), tdg("$0.016"), tdg("$10"),    tdg("No"), tdg("Yes"), td("Cheapest US")],
    [tdb("SignalWire 🇺🇸"), tdc("$1.15"), tdg("$0.018"), tdg("$5"),     tdg("No"), tdg("Yes"), td("In .env")],
    [tdb("Plivo 🇺🇸"),     tdc("$0.80"), tdg("$0.018"), tdg("$2"),     tdg("No"), tdg("Yes"), tdr("Login issues")],
    [tdb("Vonage 🇺🇸"),    tdc("$1.40"), tdo("$0.025"), tdg("€2"),     tdg("No"), tdg("Yes"), td("Not tried")],
]
t = Table(data, colWidths=[32*mm, 22*mm, 22*mm, 22*mm, 18*mm, 22*mm, 32*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "All US providers are pure pay-as-you-go — no monthly plan, no commitment. "
    "Just add credit and pay per minute used. Best for testing and low volume.",
    NOTE))

story.append(Spacer(1, 8))
story.append(section_box("Indian Providers — Monthly & Annual Plans (4× Cheaper per Minute)", LIGHT_GREEN, BORDER_GRN, GREEN))
story.append(Spacer(1, 6))

data = [
    [th("Provider"), th("Starter/mo"), th("Annual Plan"), th("Annual\nSaving"), th("India/min"), th("KYC?"), th("Status")],
    [tdb("Exotel 🇮🇳"),    tdo("₹2,499 (~$30)"), tdo("₹24,990/yr"), tdg("Save 17%"), tdg("₹0.50"), tdr("Yes"), tdr("KYC failed")],
    [tdb("Ozonetel 🇮🇳"),  tdo("₹3,000 (~$36)"), tdo("₹30,600/yr"), tdg("Save 15%"), tdg("₹0.50"), tdr("Yes"), td("Not tried")],
    [tdb("Servetel 🇮🇳"),  tdo("₹1,999 (~$24)"), tdo("₹19,990/yr"), tdg("Save 17%"), tdg("₹0.45"), tdr("Yes"), tdr("No free")],
    [tdb("MyOperator 🇮🇳"),tdo("₹3,000 (~$36)"), tdo("₹28,800/yr"), tdg("Save 20%"), tdo("₹0.55"), tdr("Yes"), td("Not tried")],
    [tdb("Knowlarity 🇮🇳"),tdo("₹3,500 (~$42)"), tdo("₹33,600/yr"), tdg("Save 20%"), tdg("₹0.50"), tdr("Yes"), td("Not tried")],
    [tdb("Airtel IQ 🇮🇳"), tdo("₹5,000 (~$60)"), tdo("₹48,000/yr"), tdg("Save 20%"), tdg("₹0.40"), tdr("Yes"), td("Enterprise")],
]
t = Table(data, colWidths=[30*mm, 30*mm, 30*mm, 22*mm, 22*mm, 18*mm, 18*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Indian providers require TRAI compliance (KYC): company GST, PAN, business registration, "
    "DLT registration. Approval takes 3–7 days. Cannot skip — Indian government regulation.",
    NOTE))

story.append(PageBreak())

# ─── SECTION 5: PLAN DETAILS ─────────────────────────────────────────────────
story.append(Paragraph("5. Exotel Plan Details (Recommended Indian Provider)", H1))
story.append(divider())

data = [
    [th("Plan"),       th("Monthly"),         th("Annual"),          th("Included\nMins/mo"), th("Overage\nRate"), th("Numbers"), th("Best For")],
    [tdb("Starter"),   tdo("₹2,499 (~$30)"),  tdo("₹24,990 (~$300)"),tdc("500"),   tdc("₹0.50"), tdc("1"), td("Testing &lt;500 calls/mo")],
    [tdb("Growth"),    tdo("₹4,999 (~$60)"),  tdo("₹47,990 (~$576)"),tdc("1,500"),tdc("₹0.48"), tdc("2"), td("1,000–3,000 calls/mo")],
    [tdb("Business"),  tdo("₹9,999 (~$120)"), tdo("₹95,990 (~$1,152)"),tdc("4,000"),tdc("₹0.45"),tdc("5"), td("3,000–8,000 calls/mo")],
    [tdb("Enterprise"),td("Custom"),          td("Negotiated"),       tdc("10,000+"),tdc("₹0.35"),tdc("∞"), td("8,000+ calls/mo")],
]
t = Table(data, colWidths=[24*mm, 30*mm, 34*mm, 22*mm, 22*mm, 20*mm, 38*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Annual plans save 15–20% over monthly. Example: Exotel Growth monthly = ₹4,999 × 12 = ₹59,988. "
    "Annual = ₹47,990. Saving = ₹11,998 (~$144/year) just by paying upfront.",
    BODY))

# ─── SECTION 6: COST COMPARISON ──────────────────────────────────────────────
story.append(Paragraph("6. Monthly Cost Comparison — 1,000 Calls/Month (2 min avg = 2,000 min)", H1))
story.append(divider())

data = [
    [th("Provider"), th("Type"), th("Number\nFee/mo"), th("2,000 min\nCost"), th("Total/mo"), th("Total/yr"), th("vs Twilio")],
    [tdb("Twilio"),    td("US Pay/use"),   tdc("$1.15"),  tdc("$44.00"),  tdo("$45.15"), tdo("$542"),   tdc("—")],
    [tdb("Telnyx"),    td("US Pay/use"),   tdc("$1.00"),  tdc("$32.00"),  tdg("$33.00"), tdg("$396"),   tdg("-$146/yr")],
    [tdb("SignalWire"),td("US Pay/use"),   tdc("$1.15"),  tdc("$36.00"),  tdo("$37.15"), tdo("$446"),   tdg("-$96/yr")],
    [tdb("Plivo"),     td("US Pay/use"),   tdc("$0.80"),  tdc("$36.00"),  tdo("$36.80"), tdo("$442"),   tdg("-$100/yr")],
    [tdb("Exotel"),    td("India Monthly"),tdc("Included"),tdc("₹500 extra"),tdo("~$36"), tdo("~$432"),  tdg("-$110/yr")],
    [tdb("Exotel"),    td("India Annual"), tdc("Included"),tdc("₹500 extra"),tdg("~$30"), tdg("~$360"),  tdg("-$182/yr")],
]
t = Table(data, colWidths=[28*mm, 28*mm, 22*mm, 22*mm, 22*mm, 22*mm, 26*mm])
t.setStyle(BASE_TS)
t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
    ("BACKGROUND",    (0,6), (-1,6), LIGHT_GREEN),
    ("GRID",          (0,0), (-1,-1), 0.5, GRAY_LIGHT),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph("* Exotel Growth plan includes 1,500 min/mo free. Extra 500 min = 500 × ₹0.48 = ₹240 (~$3). Prices indicative.", NOTE))

story.append(PageBreak())

# ─── SECTION 7: US vs INDIA ──────────────────────────────────────────────────
story.append(Paragraph("7. US vs Indian Providers — Key Differences", H1))
story.append(divider())

data = [
    [th("Factor"), th("🇺🇸 US Providers"), th("🇮🇳 Indian Providers")],
    [td("Setup time"),       tdg("5 minutes — card only"),  tdr("3–7 days — KYC + approval")],
    [td("KYC required"),     tdg("No"),                     tdr("Yes — GST, PAN, business docs")],
    [td("Call rate India"),  tdr("$0.016–0.025/min"),       tdg("$0.004–0.007/min (4× cheaper)")],
    [td("Monthly minimum"),  tdg("None — pay as you go"),   tdo("₹2,000–5,000/mo base plan")],
    [td("Caller ID shown"),  tdr("+1 US number (foreign)"), tdg("+91 Indian number (local)")],
    [td("Hospital pickup"),  tdr("Lower — looks foreign"),  tdg("Higher — local number trusted")],
    [td("Call quality"),     tdo("International routing"),  tdg("Local routing — better quality")],
    [td("Annual plans"),     tdr("Not available"),          tdg("Yes — 15–20% discount")],
    [td("WebSocket/Stream"), tdg("Full support"),           tdo("Exotel, Airtel IQ support it")],
    [td("Free trial"),       tdg("Yes — $2 to $14 credit"),tdr("None")],
    [td("Best for"),         tdo("Testing / quick start"),  tdg("Production at scale")],
]
t = Table(data, colWidths=[50*mm, 65*mm, 55*mm])
t.setStyle(BASE_TS)
story.append(t)
story.append(Spacer(1, 8))

# ─── SECTION 8: RECOMMENDATION ───────────────────────────────────────────────
story.append(Paragraph("8. Recommendation", H1))
story.append(divider())

rec_data = [
    [Paragraph("<b>Stage</b>", sty("X", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE)),
     Paragraph("<b>Action</b>", sty("X", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE)),
     Paragraph("<b>Cost</b>", sty("X", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE)),
     Paragraph("<b>Why</b>", sty("X", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE))],
    [Paragraph("Right Now", sty("X",fontSize=10,fontName="Helvetica-Bold",textColor=BLUE)),
     td("Upgrade Twilio trial → paid (add $50 credit)"),
     tdc("$50 one-time"),
     td("Removes 'press any key' message. 5 minutes to set up.")],
    [Paragraph("Short Term", sty("X",fontSize=10,fontName="Helvetica-Bold",textColor=ORANGE)),
     td("Try Telnyx ($10 free credit) — same code, just .env change"),
     tdc("$10 trial"),
     td("Cheapest US rate ($0.016/min). No KYC. Easy switch.")],
    [Paragraph("Production", sty("X",fontSize=10,fontName="Helvetica-Bold",textColor=GREEN)),
     td("Exotel Growth Annual — after Forage AI KYC complete"),
     tdc("₹47,990/yr (~$576)"),
     td("Indian number = better pickup rate. 4× cheaper per minute.")],
    [Paragraph("GPU Scale", sty("X",fontSize=10,fontName="Helvetica-Bold",textColor=PURPLE)),
     td("Runpod RTX 3080 during calling hours (9am–6pm IST)"),
     tdc("~$1.80/day (~$54/mo)"),
     td("Drops latency from ~1s to ~300ms. True production grade.")],
]
t = Table(rec_data, colWidths=[28*mm, 62*mm, 30*mm, 50*mm])
t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  BLUE),
    ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
    ("GRID",          (0,0), (-1,-1), 0.5, GRAY_LIGHT),
    ("TOPPADDING",    (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 8))

story.append(section_box(
    "Switching from Twilio to any other provider = changing 3 lines in .env file. "
    "No code changes required.",
    LIGHT_GREEN, BORDER_GRN, GREEN
))

story.append(Spacer(1, 6))

# ─── SECTION 9: BUGS FIXED ───────────────────────────────────────────────────
story.append(Paragraph("9. Bugs Fixed in Day 2", H1))
story.append(divider())

bugs = [
    ("Dr. Dr. John in greeting",          "_clean_name() strips existing Dr. prefix in prompts.py"),
    ("Echo — agent heard itself",          "speaking flag blocks incoming audio while TTS plays"),
    ("Cardiology saved as branch",         "Specialization blacklist in save_branch()"),
    ("Oh → agent says goodbye",            "Check save_branch return value before ending call"),
    ("Nothing over saved as branch",       "_NEGATIVE_WORDS check — any negative word = no branch"),
    ("10-second cold-start latency",       "Pre-warm Whisper + Piper before placing call"),
    ("23-second LLM wait",                 "Expanded heuristics + 5s LLM timeout + fallback re-ask"),
    ("Are you talking about Dr. John?",    "Added to _NAME_CONFIRMATION_PHRASES — heuristic catches it"),
    ("Dallas transcribed as Ballad",       "Added Indian city names as initial_prompt to Whisper"),
    ("Which branch? from caller",          "New _CLARIFICATION_PHRASES heuristic response"),
    ("TTS delay on every turn",            "Pre-cache all fixed responses at session start — 0ms"),
    ("Silence cuts too early",             "_SILENCE_FRAMES reduced 50→25 (1000ms→500ms)"),
]
data = [[th("Bug"), th("Fix")]]
for b, f in bugs:
    data.append([tdb(b), td(f)])
t = Table(data, colWidths=[80*mm, 90*mm])
t.setStyle(BASE_TS)
story.append(t)

story.append(Spacer(1, 10))
story.append(Paragraph(
    "All code is in C:\\Users\\salom\\OneDrive\\Desktop\\conversational_ai\\   "
    "·   Run with: python run_twilio.py --doctor 'Name' --hospital 'Apollo' --to '+91XXXXXXXXXX'",
    NOTE))

# ── Build ─────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=18*mm,  bottomMargin=18*mm,
)
doc.build(story, onFirstPage=cover_page, onLaterPages=later_page)
print(f"PDF saved: {OUTPUT}")
