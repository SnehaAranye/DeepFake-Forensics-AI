
import os
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
)
import matplotlib.pyplot as plt
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def create_simple_pdf(
    title, filename, summary_rows, headers, rows, note,
    images=None, extra_sections=None, extra_tables=None
):
    buffer = BytesIO()
    image_buffers = []

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    content = [
        Paragraph(escape(title), styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            f"<b>Report generated:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    content.append(Paragraph("Analysis Summary", styles["Heading2"]))

    for label, value in summary_rows:
        content.append(
            Paragraph(
                f"<b>{escape(str(label))}:</b> "
                f"{escape(str(value))}",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 5))

    content.append(Spacer(1, 12))
    content.append(Paragraph("Detailed Scores", styles["Heading2"]))

    table_data = [
        [Paragraph(escape(str(h)), styles["Normal"]) for h in headers]
    ]

    for row in rows:
        table_data.append([
            Paragraph(escape(str(value)), styles["Normal"])
            for value in row
        ])

    col_width = 7.3 * inch / max(1, len(headers))
    table = Table(
        table_data,
        colWidths=[col_width] * len(headers),
        repeatRows=1,
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    content.append(table)
    content.append(Spacer(1, 14))

    if extra_sections:
        for heading, text_value in extra_sections:
            content.append(Paragraph(escape(heading), styles["Heading2"]))
            content.append(
                Paragraph(escape(str(text_value)), styles["Normal"])
            )
            content.append(Spacer(1, 10))

    if extra_tables:
        for heading, table_headers, table_rows in extra_tables:
            content.append(
                Paragraph(escape(heading), styles["Heading2"])
            )

            extra_data = [
                [
                    Paragraph(escape(str(item)), styles["Normal"])
                    for item in table_headers
                ]
            ]

            for row in table_rows:
                extra_data.append([
                    Paragraph(escape(str(item)), styles["Normal"])
                    for item in row
                ])

            extra_width = 7.3 * inch / max(1, len(table_headers))
            extra_table = Table(
                extra_data,
                colWidths=[extra_width] * len(table_headers),
                repeatRows=1,
            )
            extra_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))

            content.append(extra_table)
            content.append(Spacer(1, 12))

    if images:
        content.append(Paragraph("Visual Explanations", styles["Heading2"]))

        for caption, pil_image in images:
            if pil_image is None:
                continue

            image_buffer = BytesIO()
            pil_image.convert("RGB").save(image_buffer, format="PNG")
            image_buffer.seek(0)
            image_buffers.append(image_buffer)

            img_width, img_height = pil_image.size
            scale = min(
                (6.5 * inch) / max(1, img_width),
                (4.5 * inch) / max(1, img_height),
            )

            content.append(Spacer(1, 8))
            content.append(
                Paragraph(escape(caption), styles["Heading3"])
            )
            content.append(
                PDFImage(
                    image_buffer,
                    width=img_width * scale,
                    height=img_height * scale,
                )
            )
            content.append(Spacer(1, 12))

    content.append(Paragraph("Important Limitation", styles["Heading2"]))
    content.append(Paragraph(escape(note), styles["Normal"]))

    doc.build(content)
    return buffer.getvalue()


# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="DeepFake Forensics AI",
    page_icon="🔍",
    layout="wide",
)

MODEL_ID = "sakshamkr1/deitfake-v2"
DEVICE = torch.device("cpu")


# -------------------- LOAD VISION TRANSFORMER --------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    labels = {
        int(index): str(label)
        for index, label in model.config.id2label.items()
    }

    fake_indices = [
        index for index, label in labels.items()
        if "fake" in label.lower()
    ]

    real_indices = [
        index for index, label in labels.items()
        if "real" in label.lower()
    ]

    if not fake_indices or not real_indices:
        raise ValueError(
            f"Cannot map model labels safely: {labels}"
        )

    return processor, model, labels, fake_indices[0], real_indices[0]


def predict_image(image, processor, model):
    """Return the model probabilities for one image."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    return probabilities.cpu().numpy()
# -------------------- AUDIO DEEPFAKE DETECTION --------------------


AUDIO_MODEL_ID = "Hemgg/Deepfake-audio-detection"


@st.cache_resource
def load_audio_model():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        AUDIO_MODEL_ID
    )

    model = AutoModelForAudioClassification.from_pretrained(
        AUDIO_MODEL_ID
    )

    model.to(DEVICE)
    model.eval()

    labels = {
        int(index): str(label)
        for index, label in model.config.id2label.items()
    }

    return feature_extractor, model, labels


def predict_audio_file(audio_path, feature_extractor, model):
    waveform, _ = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
        duration=20
    )

    if waveform.size == 0:
        raise ValueError("The uploaded audio is empty.")

    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    return probabilities.cpu().numpy()


def explain_audio_segments(audio_path, feature_extractor, model, labels, segment_seconds=1.0):
    waveform, sr = librosa.load(
        audio_path, sr=16000, mono=True, duration=20
    )

    if waveform.size == 0:
        raise ValueError("The uploaded audio is empty.")

    def get_scores(audio):
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    baseline_scores = get_scores(waveform)
    predicted_index = int(np.argmax(baseline_scores))
    baseline_score = float(baseline_scores[predicted_index])

    segment_samples = int(segment_seconds * sr)
    sensitivity = []

    for start in range(0, len(waveform), segment_samples):
        end = min(start + segment_samples, len(waveform))
        masked = waveform.copy()
        masked[start:end] = 0.0

        masked_scores = get_scores(masked)
        change = abs(
            baseline_score - float(masked_scores[predicted_index])
        )

        sensitivity.append({
            "Start (seconds)": start / sr,
            "End (seconds)": end / sr,
            "Score change": change
        })

    return waveform, sr, pd.DataFrame(sensitivity), labels[predicted_index], baseline_score


# -------------------- TEMPORAL ANALYSIS --------------------

def extract_video_frames(video_path, max_frames=12):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Could not open this video file.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        capture.release()
        raise ValueError("The video contains no readable frames.")

    if not fps or fps <= 0:
        fps = 1.0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        capture.release()
        raise RuntimeError("OpenCV face detector could not be loaded.")

    indices = np.linspace(
        0,
        total_frames - 1,
        min(max_frames, total_frames),
        dtype=int,
    )

    results = []

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # Skip frames where this face detector finds no face.
        if len(faces) == 0:
            continue

        # Select the largest detected face.
        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3],
        )

        # Add some context around the detected face.
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_rgb = frame_rgb[y1:y2, x1:x2]

        if face_rgb.size == 0:
            continue

        results.append({
            "frame_number": int(index),
            "time_seconds": round(float(index) / fps, 2),
            "image": Image.fromarray(face_rgb).resize((224, 224)),
            "preview": Image.fromarray(frame_rgb),
        })

    capture.release()

    if not results:
        raise ValueError(
            "No faces were detected in the sampled frames. "
            "Try a video with a clearly visible, front-facing face. "
            "This does not mean the video is real."
        )

    return results


# -------------------- EXPLAINABLE AI --------------------
def create_occlusion_explanation(
    image,
    processor,
    model,
    target_index,
    grid_size=5,
):
    """
    Occlude image patches and measure how the target-class
    probability changes. This is an approximate XAI method.
    """
    image = image.resize((224, 224)).convert("RGB")
    original = np.array(image)
    height, width = original.shape[:2]

    occluded_images = []

    for row in range(grid_size):
        for col in range(grid_size):
            altered = original.copy()

            y1 = row * height // grid_size
            y2 = (row + 1) * height // grid_size
            x1 = col * width // grid_size
            x2 = (col + 1) * width // grid_size

            # Replace a patch with neutral gray.
            altered[y1:y2, x1:x2] = 127
            occluded_images.append(Image.fromarray(altered))

    changes = []

    # Small batches help limit CPU memory use.
    for start in range(0, len(occluded_images), 4):
        batch = occluded_images[start:start + 4]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            batch_scores = probabilities[:, target_index].cpu().numpy()

        changes.extend(batch_scores.tolist())

    original_probability = float(
        predict_image(image, processor, model)[target_index]
    )

    # Positive values indicate patches whose removal lowered
    # the model's confidence in the selected class.
    explanation_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for position, occluded_probability in enumerate(changes):
        row = position // grid_size
        col = position % grid_size
        explanation_grid[row, col] = (
            original_probability - occluded_probability
        )

    # Scale the coarse explanation grid for visualization.
    explanation_grid = cv2.resize(
        explanation_grid,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    minimum = float(explanation_grid.min())
    maximum = float(explanation_grid.max())

    if maximum - minimum > 1e-8:
        normalized = (explanation_grid - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(explanation_grid)

    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

    return Image.fromarray(overlay)


# -------------------- USER INTERFACE --------------------


# ============== PURPLE AI FORENSICS DASHBOARD ==============

import streamlit as st

# ---------- PREMIUM DARK THEME ----------
st.markdown("""
<style>
:root {
    color-scheme: dark;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 5% 0%, #241747 0%, transparent 38%),
        radial-gradient(ellipse at 100% 25%, #20113D 0%, transparent 32%),
        #0C0B14;
    color: #E9E5F5;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding: 1.8rem 2.3rem 3rem;
}

/* GLOBAL TEXT CONTRAST */
.stApp,
.stApp p,
.stApp label,
.stApp li,
.stApp span,
.stApp small,
.stApp .stMarkdown,
[data-testid="stCaptionContainer"] {
    color: #D5D0E5;
}

h1, h2, h3, h4 {
    color: #F6F2FF !important;
    letter-spacing: -0.4px;
}

[data-testid="stSidebar"] {
    background: #11101B;
    border-right: 1px solid #302747;
}

[data-testid="stSidebar"] * {
    color: #E3DDF3;
}

/* SIDEBAR NAVIGATION */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.7;
}

/* HERO BANNER */
.forensic-hero {
    position: relative;
    overflow: hidden;
    padding: 38px 36px;
    margin: 8px 0 28px;
    border-radius: 24px;
    border: 1px solid #59418C;
    background: linear-gradient(
        120deg,
        #211638 0%,
        #39205D 48%,
        #512D80 100%
    );
    box-shadow: 0 15px 45px rgba(106, 65, 190, 0.15);
    animation: heroEnter 0.7s ease-out both;
}

.forensic-hero:after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -65px;
    top: -100px;
    border-radius: 50%;
    border: 1px solid rgba(222, 194, 255, 0.18);
    box-shadow:
        0 0 0 25px rgba(222,194,255,0.035),
        0 0 0 55px rgba(222,194,255,0.025);
    pointer-events: none;
}

.hero-kicker {
    display: inline-block;
    color: #DCC5FF !important;
    border: 1px solid #8062B2;
    background: rgba(24, 13, 43, 0.35);
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.7px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: clamp(30px, 4vw, 46px);
    line-height: 1.15;
    font-weight: 850;
    margin: 19px 0 12px;
    text-align: center;
    animation: titleAnimation 2s ease-in-out infinite;
}
@keyframes titleAnimation {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.04);
    }
    100% {
        transform: scale(1);
    }
}
.hero-description {
    color: #E0D9EF !important;
    max-width: 800px;
    font-size: 15px;
    line-height: 1.85;
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 20px;
}

.hero-chip {
    display: inline-block;
    color: #F1E8FF !important;
    background: rgba(16, 9, 29, 0.35);
    border: 1px solid #73569E;
    border-radius: 9px;
    padding: 8px 11px;
    font-size: 11px;
    font-weight: 700;
}

/* SECTION LABELS */
.section-kicker {
    color: #C5A1FF !important;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 2px;
    margin: 22px 0 7px;
}

/* DASHBOARD CARDS */
.forensic-card {
    height: 100%;
    min-height: 245px;
    box-sizing: border-box;
    background: linear-gradient(145deg, #191625, #151320);
    border: 1px solid #332A48;
    border-radius: 19px;
    padding: 23px 21px;
    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
    animation: cardEnter 0.65s ease-out both;
}

.forensic-card:hover {
    transform: translateY(-6px);
    border-color: #9C72E8;
    box-shadow: 0 12px 35px rgba(128, 82, 208, 0.16);
}

.card-icon {
    font-size: 28px;
    margin-bottom: 16px;
}

.card-kicker {
    color: #BE9CFA !important;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1.3px;
    margin-bottom: 10px;
}

.card-title {
    color: #F6F2FF !important;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 10px;
}

.card-description {
    color: #C5BED7 !important;
    font-size: 13px;
    line-height: 1.8;
    min-height: 68px;
}

.card-format {
    color: #D8C4FF !important;
    background: #241B36;
    border: 1px solid #493765;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 14px;
    font-size: 10px;
    font-weight: 750;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: #171421;
    border: 1px solid #302642;
    border-radius: 15px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #C5BED7 !important;
}

[data-testid="stMetricValue"] {
    color: #F3EAFF !important;
}

/* UPLOAD AREA: HIGH CONTRAST FIX */
[data-testid="stFileUploader"] {
    background: #171421 !important;
    border: 1px solid #423457 !important;
    border-radius: 15px !important;
    padding: 12px !important;
    overflow: hidden;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"] {
    background: #1B1728 !important;
    border-color: #6D5290 !important;
    color: #EAE3F8 !important;
}

[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * {
    color: #EAE3F8 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #EAE3F8 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #D6CBEA !important;
}

[data-testid="stFileUploader"] button {
    background: #7547B8 !important;
    color: #FFFFFF !important;
    border: 1px solid #A47BDF !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #8A5BD0 !important;
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
    background: #1B1728 !important;
    color: #F1ECFA !important;
    border-color: #493A60 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #AAA0BE !important;
}

/* BUTTONS */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #7547B8, #9565D8) !important;
    color: #FFFFFF !important;
    border: 1px solid #A783E0 !important;
    border-radius: 10px !important;
    min-height: 43px;
    font-weight: 750 !important;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 22px rgba(147, 98, 218, 0.25);
}

/* TABS, EXPANDERS, TABLES */
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #171421 !important;
    border-color: #39304A !important;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #EAE3F8 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #D5CBE8 !important;
}

.stTabs [aria-selected="true"] {
    color: #D8B9FF !important;
}

hr {
    border-color: #342A46 !important;
}

/* NOTICE */
.forensic-notice {
    padding: 16px 18px;
    margin: 25px 0;
    background: #191426;
    border: 1px solid #47365F;
    border-left: 4px solid #A77BE8;
    border-radius: 12px;
    color: #D5CBE8 !important;
    font-size: 12px;
    line-height: 1.8;
}

.forensic-notice strong {
    color: #E3CFFF !important;
}

/* SUBTLE ANIMATIONS */
@keyframes heroEnter {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes cardEnter {
    from { opacity: 0; transform: translateY(9px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    .forensic-hero,
    .forensic-card {
        animation: none;
        transition: none;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .forensic-hero {
        padding: 25px 20px;
    }

    .forensic-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)




# ---------- HERO ----------
st.markdown("""
<div class="forensic-hero">
    <div class="hero-kicker">AI SECURITY LAB · DIGITAL FORENSICS</div>
    <div class="hero-title">DeepFake Detection  &amp; Forensics</div>
    <div class="hero-description">
        Investigate digital media with AI-assisted image inspection,
        audio analysis, temporal video review, and explainable predictions.
        A clean workspace built for a more transparent forensic workflow.
    </div>
    <div class="hero-chips">
        <span class="hero-chip">✦ Vision Transformer</span>
        <span class="hero-chip">◷ Temporal Analysis</span>
        <span class="hero-chip">◎ Explainable AI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------- QUICK STATUS ----------
st.markdown(
    '<div class="section-kicker">WORKSPACE OVERVIEW</div>',
    unsafe_allow_html=True
)
st.markdown("## Your forensic workspace")

metric1, metric2  = st.columns(2)

with metric1:
    st.metric("Analysis modes", "03", help="Image, audio, and video")

with metric2:
    st.metric("Upload limit", "200 MB", help="Configure Streamlit's actual upload limit separately")




# ---------- ANALYSIS MODE CARDS ----------
st.markdown(
    '<div class="section-kicker">CHOOSE YOUR INVESTIGATION</div>',
    unsafe_allow_html=True
)
st.markdown("## Select a media type")

image_col, audio_col, video_col = st.columns(3, gap="medium")

with image_col:
    st.markdown("""
    <div class="forensic-card">
        <div class="card-icon">▧</div>
        <div class="card-kicker">01 / VISUAL INSPECTION</div>
        <div class="card-title">Image Analysis</div>
        <div class="card-description">
            Inspect images and faces, review model confidence,
            and explore approximate sensitivity heatmaps.
        </div>
        <div class="card-format">JPG · JPEG · PNG · WEBP</div>
    </div>
    """, unsafe_allow_html=True)

with audio_col:
    st.markdown("""
    <div class="forensic-card">
        <div class="card-icon">≋</div>
        <div class="card-kicker">02 / AUDIO INSPECTION</div>
        <div class="card-title">Audio Analysis</div>
        <div class="card-description">
            Examine audio predictions, waveform patterns,
            and supported segment-level explanations.
        </div>
        <div class="card-format">WAV · MP3 · FLAC · OGG · M4A</div>
    </div>
    """, unsafe_allow_html=True)

with video_col:
    st.markdown("""
    <div class="forensic-card">
        <div class="card-icon">▷</div>
        <div class="card-kicker">03 / TEMPORAL INSPECTION</div>
        <div class="card-title">Video Forensics</div>
        <div class="card-description">
            Review sampled frames, compare scores over time,
            and explore supported visual explanations.
        </div>
        <div class="card-format">MP4 · AVI · MOV · MKV</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- TRUST & SAFETY NOTE ----------
st.markdown("""
<div class="forensic-notice">
    <strong>Important forensic note</strong><br>
    Model predictions, confidence scores, and visual explanations are
    estimates. They are not definitive proof that a file is genuine
    or manipulated. Results depend on the trained model and input quality.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Keep your existing image, audio, and video analysis code below.
# Do not delete or replace your prediction functions.

# ============== END PURPLE DASHBOARD UI ==============



# -------------------- IMAGE ANALYSIS --------------------

def generate_face_heatmap(image, processor, model, fake_index):
    image = image.convert("RGB")
    original = np.array(image)
    height, width = original.shape[:2]

    patch_size = 20
    stride = 12
    sensitivity = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    def get_fake_score(img):
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[fake_index].item())

    baseline = get_fake_score(image)
    fill = tuple(int(v) for v in original.mean(axis=(0, 1)))

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y2 = min(y + patch_size, height)
            x2 = min(x + patch_size, width)

            masked = original.copy()
            masked[y:y2, x:x2] = fill

            score = get_fake_score(Image.fromarray(masked))
            change = abs(baseline - score)

            sensitivity[y:y2, x:x2] += change
            counts[y:y2, x:x2] += 1

    sensitivity /= np.maximum(counts, 1)
    maximum = float(sensitivity.max())

    if maximum > 0:
        heat = np.uint8(255 * sensitivity / maximum)
    else:
        heat = np.zeros((height, width), dtype=np.uint8)

    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.55, colored, 0.45, 0)

    return Image.fromarray(overlay), Image.fromarray(colored)

st.header("1. Image DeepFake Analysis")

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image_upload"
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Original uploaded image", width=400)

    if st.button("Analyze Image", key="analyze_image"):
        try:
            with st.spinner("Detecting face and analyzing image..."):
                # Detect a face
                image_array = np.array(image)
                gray = cv2.cvtColor(
                    image_array, cv2.COLOR_RGB2GRAY
                )

                face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )

                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                if len(faces) > 0:
                    # Select the largest detected face
                    x, y, w, h = max(
                        faces, key=lambda face: face[2] * face[3]
                    )

                    face_image = image.crop(
                        (x, y, x + w, y + h)
                    )
                    st.write("Original image size:", image.size)
                    st.write("Detected face box:", (x, y, w, h))
                    st.write("Face crop size:", face_image.size)
                    st.image(face_image, caption="Face sent to model", width=300)

                    st.image(
                        face_image,
                        caption="Face selected for analysis",
                        width=300
                    )

                    image_to_analyze = face_image

                else:
                    st.warning(
                        "No face detected. The prediction may be "
                        "unreliable because this model is designed "
                        "for facial images."
                    )
                    image_to_analyze = image

                processor, model, labels, fake_index, real_index = load_model()

                probabilities = predict_image(
                    image_to_analyze, processor, model
                )

                fake_score = float(probabilities[fake_index])
                real_score = float(probabilities[real_index])
                predicted_index = int(np.argmax(probabilities))
                predicted_label = labels[predicted_index]

            st.divider()
            st.subheader("🔎 Image Analysis Report")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake-class score", f"{fake_score:.2%}")
            with col2:
                st.metric("Real-class score", f"{real_score:.2%}")
            st.write("### 🧠 AI Prediction")
            if predicted_label.lower() == "fake":
                st.warning(
                    "The AI model flagged this image as potentially manipulated.")
            else:
                st.success(
                    "The AI model classified this image as likely authentic.")
            st.write("### 📊 Model Score Comparison")
            st.progress(float(fake_score),
                        text=f"Fake class: {fake_score:.2%}")
            st.progress(float(real_score),
                        text=f"Real class: {real_score:.2%}")
            with st.expander("🔍 How did the AI decide?"):
                st.write(
                    "The model compares the face image with patterns learned during "
                    "training and produces scores for its Fake and Real classes."
                    )
                st.write(
                        "These scores do not prove whether an image is genuine or manipulated."
                        )
                st.divider()
                st.subheader("🧠 Visual Explainable AI")
                with st.spinner("Generating face heatmap..."):
                    explanation, heatmap = generate_face_heatmap(
                    image_to_analyze,
                    processor,
                    model,
                    fake_index
                    )
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        explanation,
                        caption="Face with sensitivity overlay",
                        use_container_width=True
                        )
                with col2:
                    st.image(
                        heatmap,
                        caption="Model sensitivity heatmap",
                        use_container_width=True
                        )
                with st.expander("How to understand this heatmap"):
                    st.write(
                        "Colored regions show where masking image patches changed the "
                        "model's Fake-class score. This helps explain model sensitivity; "
                        "it does not prove that the image is fake."
                        )

            image_pdf = create_simple_pdf(
                "DeepFake Image Analysis Report",
                "image_report.pdf",
                [
                    ("Uploaded file", uploaded_image.name),
                    ("Analysis date", datetime.now().strftime("%d-%m-%Y %H:%M")),
                    ("Prediction", predicted_label),
                    ("Fake-class score", f"{fake_score:.2%}"),
                    ("Real-class score", f"{real_score:.2%}"),
                    ("Face detected", "Yes" if len(faces) > 0 else "No"),
                    ("Model", MODEL_ID),
                ],
                ["Class", "Model score"],
                [
                    ["Fake", f"{fake_score:.2%}"],
                    ["Real", f"{real_score:.2%}"],
                    ],
                    (
                        "This report contains model scores and approximate sensitivity "
                        "visualizations. A heatmap shows regions that affected the model's "
                        "score; it does not prove that an image is manipulated. Predictions "
                        "may be unreliable when no suitable face is detected."
                    ),
                images=[
                    ("Image/face analyzed", image_to_analyze),
                    ("Face sensitivity overlay", explanation),
                    ("Model sensitivity heatmap", heatmap),
                    ],
                    extra_sections=[
                        (
                            "Method",
                            "The image is classified by a Vision Transformer image "
                            "classification model. Fake-class and Real-class scores are "
                            "the model's output probabilities.",
                        ),
                        (
                            "Explainable AI",
                            "Image patches are masked and the change in the Fake-class "
                            "score is measured. The overlay and heatmap visualize model "
                            "sensitivity, not confirmed evidence of forgery.",
                        ),
                    ],
            )

            st.download_button(
                "Download Image PDF Report",
                data=image_pdf,
                file_name="deepfake_image_report.pdf",
                mime="application/pdf",
                key="download_image_pdf",
            )

        except Exception as error:
            st.error("Image analysis failed.")
            st.exception(error)
else:
    st.warning("Upload a Image to begin the analysis.")
# -------------------- AUDIO ANALYSIS --------------------

# -------------------- AUDIO ANALYSIS --------------------

st.divider()
st.header("2. Audio DeepFake Analysis")

uploaded_audio = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    key="audio_upload",
)

if uploaded_audio is not None:
    st.audio(uploaded_audio)

    if st.button("Analyze Audio", key="analyze_audio"):
        audio_path = None

        try:
            suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_audio:
                temp_audio.write(uploaded_audio.getvalue())
                audio_path = temp_audio.name

            with st.spinner("Analyzing audio and generating explanation..."):
                processor, model, audio_labels = load_audio_model()

                probabilities = predict_audio_file(
                    audio_path, processor, model
                )

                ranking = np.argsort(probabilities)[::-1]
                top_index = int(ranking[0])
                top_label = audio_labels[top_index]
                top_score = float(probabilities[top_index])

                score_data = [
                    {
                        "Model label": audio_labels[int(index)],
                        "Score": float(probabilities[index]),
                    }
                    for index in ranking
                ]

                st.subheader("Audio classification scores")
                st.dataframe(
                    pd.DataFrame(score_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.metric(
                    "Highest-scoring class",
                    top_label,
                    f"{top_score:.1%}",
                )

                normalized = {
                    index: label.lower().replace("-", "_").replace(" ", "_")
                    for index, label in audio_labels.items()
                }

                fake_indices = [
                    i for i, label in normalized.items()
                    if "fake" in label or "spoof" in label
                    or "synthetic" in label
                ]

                real_indices = [
                    i for i, label in normalized.items()
                    if "real" in label or "bonafide" in label
                    or "bona_fide" in label or "genuine" in label
                ]

                interpretation = (
                    "Could not reliably map the model labels to real/fake"
                )
                fake_score = None
                real_score = None

                if fake_indices and real_indices:
                    fake_score = sum(
                        float(probabilities[i]) for i in fake_indices
                    )
                    real_score = sum(
                        float(probabilities[i]) for i in real_indices
                    )
                    interpretation = (
                        "Fake-like audio"
                        if fake_score > real_score
                        else "Real-like audio"
                    )

                st.write("Interpretation:", interpretation)

                if fake_score is not None:
                    col1, col2 = st.columns(2)
                    col1.metric("Fake-like score", f"{fake_score:.2%}")
                    col2.metric("Real-like score", f"{real_score:.2%}")

                # Generate audio XAI in this same analysis run.
                waveform, sr, sensitivity_df, predicted_class, base_score = (
                    explain_audio_segments(
                        audio_path,
                        processor,
                        model,
                        audio_labels,
                    )
                )

                duration = len(waveform) / sr

                st.subheader("Audio Explainable AI")
                st.write(f"Analyzed duration: {duration:.2f} seconds")
                st.write(f"Explanation class: {predicted_class}")
                st.write(f"Explanation score: {base_score:.2%}")

                # Create a waveform plot for the PDF.
                waveform_buffer = BytesIO()
                fig, ax = plt.subplots(figsize=(9, 3))
                step = max(1, len(waveform) // 8000)
                waveform_times = np.arange(
                    0, len(waveform), step
                ) / sr

                ax.plot(waveform_times, waveform[::step], linewidth=0.6)
                ax.set_title("Audio waveform")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                fig.tight_layout()
                fig.savefig(
                    waveform_buffer,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                waveform_buffer.seek(0)
                waveform_plot = Image.open(waveform_buffer).copy()

                # Create a segment-sensitivity plot for the PDF.
                sensitivity_buffer = BytesIO()
                fig, ax = plt.subplots(figsize=(9, 3))

                ax.bar(
                    sensitivity_df["Start (seconds)"],
                    sensitivity_df["Score change"],
                    width=0.8,
                )
                ax.set_title("Audio segment sensitivity")
                ax.set_xlabel("Segment start time (seconds)")
                ax.set_ylabel("Change in selected-class score")
                fig.tight_layout()
                fig.savefig(
                    sensitivity_buffer,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                sensitivity_buffer.seek(0)
                sensitivity_plot = Image.open(sensitivity_buffer).copy()

                st.write("### Audio waveform")
                st.image(waveform_plot, use_container_width=True)

                st.write("### Which segments affect the score?")
                st.image(sensitivity_plot, use_container_width=True)
                st.dataframe(
                    sensitivity_df,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("How to understand this explanation"):
                    st.write(
                        "Each segment is replaced with silence and the "
                        "model is run again. A larger score change means "
                        "the selected-class score was more sensitive to "
                        "that segment. This does not prove that the voice "
                        "is fake."
                    )

                # Prepare the detailed audio PDF.
                audio_rows = [
                    [
                        item["Model label"],
                        f'{item["Score"]:.2%}',
                    ]
                    for item in score_data
                ]

                segment_rows = [
                    [
                        f'{row["Start (seconds)"]:.2f}',
                        f'{row["End (seconds)"]:.2f}',
                        f'{row["Score change"]:.5f}',
                    ]
                    for _, row in sensitivity_df.iterrows()
                ]

                audio_pdf = create_simple_pdf(
                    "DeepFake Audio Analysis Report",
                    "audio_report.pdf",
                    [
                        ("Uploaded file", uploaded_audio.name),
                        ("Model", AUDIO_MODEL_ID),
                        ("Highest-scoring class", top_label),
                        ("Highest class score", f"{top_score:.2%}"),
                        ("Interpretation", interpretation),
                        ("Analyzed duration", f"{duration:.2f} seconds"),
                        ("Explanation class", predicted_class),
                        ("Explanation score", f"{base_score:.2%}"),
                    ],
                    ["Model class", "Score"],
                    audio_rows,
                    (
                        "This model analyzes up to 20 seconds of audio. "
                        "Its result is experimental and is not proof that "
                        "a voice is genuine or fake. Segment sensitivity "
                        "measures changes in model scores, not confirmed "
                        "evidence of audio manipulation."
                    ),
                    images=[
                        ("Audio waveform", waveform_plot),
                        ("Audio segment sensitivity", sensitivity_plot),
                    ],
                    extra_tables=[
                        (
                            "Segment-by-Segment Sensitivity",
                            ["Start (seconds)", "End (seconds)",
                                     "Score change"],
                            segment_rows,
                        ),
                    ],
                    extra_sections=[
                        (
                            "Method",
                            "The audio is loaded as mono audio at 16 kHz "
                            "and analyzed using a Wav2Vec2-based audio "
                            "classification model.",
                        ),
                        (
                            "Segment sensitivity results",
                            "The following table shows the score change "
                            "after each one-second segment is replaced "
                            "with silence.",
                        ),
                    ],
                )

                # Add the detailed segment table to the report separately
                # by generating a second PDF section in the next step below.
                st.download_button(
                    "Download Detailed Audio PDF Report",
                    data=audio_pdf,
                    file_name="deepfake_audio_report.pdf",
                    mime="application/pdf",
                    key="download_audio_pdf",
                )

        except Exception as error:
            st.error("Audio analysis failed.")
            st.exception(error)

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    st.warning(
                        "The temporary audio file could not be removed."
                    )
else:
    st.warning("Upload a Audio to begin the analysis.")

# -------------------- VIDEO ANALYSIS --------------------
st.header("3. Vedio DeepFake Analysis")
uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Analyze Video", type="primary"):
        try:
            with st.spinner("Loading the Vision Transformer..."):
                processor, model, labels, fake_index, real_index = (
                    load_model()
                )

            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(uploaded_video.getvalue())
                video_path = temporary_file.name

            try:
                with st.spinner(
                    "Extracting frames and analyzing temporal scores..."
                ):
                    frames = extract_video_frames(video_path, max_frames=12)
                    records = []

                    for frame_data in frames:
                        probabilities = predict_image(
                            frame_data["image"],
                            processor,
                            model,
                        )

                        fake_probability = float(
                            probabilities[fake_index]
                        )
                        real_probability = float(
                            probabilities[real_index]
                        )

                        predicted_index = int(np.argmax(probabilities))

                        records.append({
                            "Frame": frame_data["frame_number"],
                            "Time (seconds)": frame_data["time_seconds"],
                            "Fake probability": fake_probability,
                            "Real probability": real_probability,
                            "Frame prediction": labels[predicted_index],
                            "_image": frame_data["image"],
                            "_preview": frame_data["preview"],
                            "_probabilities": probabilities,
                        })

                results_df = pd.DataFrame([
                    {
                        key: value
                        for key, value in record.items()
                        if not key.startswith("_")
                    }
                    for record in records
                ])

                mean_fake_probability = float(
                    results_df["Fake probability"].mean()
                )

                st.header("1. Video Analysis Results")

                col1, col2, col3 = st.columns(3)

                video_prediction = (
                    "Fake-like"
                    if mean_fake_probability >= 0.5
                    else "Real-like"
                )

                video_confidence = (
                    mean_fake_probability
                    if mean_fake_probability >= 0.5
                    else 1 - mean_fake_probability
                )

                col1.metric("Aggregate result", video_prediction)
                col2.metric(
                    "Mean fake-class score",
                    f"{mean_fake_probability:.1%}",
                )
                col3.metric(
                    "Frames analyzed",
                    len(records),
                )

                st.caption(
                    "The aggregate result uses the mean fake-class "
                    "probability with a 0.5 threshold. This is a prototype "
                    "decision rule, not a validated video-level classifier."
                )

                st.header("2. Temporal Analysis")

                chart = px.line(
                    results_df,
                    x="Time (seconds)",
                    y="Fake probability",
                    markers=True,
                    range_y=[0, 1],
                    title="Fake-Class Probability Across Sampled Frames",
                )

                chart.add_hline(
                    y=0.5,
                    line_dash="dash",
                    annotation_text="0.5 reference threshold",
                )

                st.plotly_chart(chart, use_container_width=True)

                st.dataframe(
                    results_df.drop(
                        columns=["Real probability"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Video Explainable AI")

                st.write(
                    "Explore which video frames have the highest fake probability "
                    "and inspect the model's sensitivity to different image regions."
                )

                # 1. Show fake probability over time
                fig = px.scatter(
                    results_df,
                    x="Time (seconds)",
                    y="Fake probability",
                    color="Fake probability",
                    hover_data=["Frame", "Frame prediction"],
                    range_y=[0, 1],
                    color_continuous_scale="RdYlGn_r",
                    title="Frame-Level Fake Probability Over Time"
                )

                fig.add_hline(
                    y=0.5,
                    line_dash="dash",
                    annotation_text="0.5 reference"
                )

                st.plotly_chart(fig, use_container_width=True)

                # 2. Explain the three highest-scoring frames
                st.subheader("Important Frames and AI Explanations")

                important_records = sorted(
                    records,
                    key=lambda item: item["Fake probability"],
                    reverse=True
                )[:3]

                for record in important_records:
                    original_frame = record["_image"]

                    explanation = create_occlusion_explanation(
                        original_frame,
                        processor,
                        model,
                        fake_index,
                        grid_size=5
                    )
                    record["_explanation"] = explanation

                    st.markdown(
                        f"**Frame {record['Frame']} — "
                        f"{record['Time (seconds)']:.2f} seconds**"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            original_frame,
                            caption="Original video frame",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            explanation,
                            caption="AI sensitivity heatmap",
                            use_container_width=True
                        )

                    st.write(
                        f"Fake probability: "
                        f"{record['Fake probability']:.2%}"
                    )

                    st.write(
                        f"Frame prediction: {record['Frame prediction']}"
                    )

                    st.divider()

                # ---------------- PDF REPORT ----------------
                # ---------------- DETAILED VIDEO PDF REPORT ---------------
                # Create a graph image for the PDF.
                temporal_buffer = BytesIO()
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(
                    results_df["Time (seconds)"],
                    results_df["Fake probability"],
                    marker="o",
                )
                ax.axhline(0.5, linestyle="--")
                ax.set_ylim(0, 1)
                ax.set_title("Fake-Class Probability Across Video Time")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Fake-class probability")
                fig.tight_layout()
                fig.savefig(
                    temporal_buffer,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                temporal_buffer.seek(0)
                temporal_plot = Image.open(temporal_buffer).copy()
                # Include the three most important frames and their heatmaps.
                video_report_images = [
                ("Fake-class probability over time", temporal_plot)
                ]
                for record in important_records:
                    video_report_images.append((
                        f"Original face crop — frame {record['Frame']} "
                        f"at {record['Time (seconds)']:.2f} seconds",
                        record["_image"],
                    ))
                    video_report_images.append((
                        f"XAI sensitivity heatmap — frame {record['Frame']}",
                        record["_explanation"],
                    ))
                video_frame_rows = [
                    [
                        str(record["Frame"]),
                        f'{record["Time (seconds)"]:.2f}',
                        f'{record["Fake probability"]:.2%}',
                        f'{record["Real probability"]:.2%}',
                        str(record["Frame prediction"]),
                    ]
                    for record in records
                ]
                video_pdf = create_simple_pdf(
                    "DeepFake Forensics AI — Video Analysis Report",
                    "video_report.pdf",
                    [
                        ("Uploaded file", uploaded_video.name),
                        ("Analysis date", datetime.now().strftime("%d-%m-%Y %H:%M")),
                        ("Model", MODEL_ID),
                        ("Overall prototype result", video_prediction),
                        ("Mean fake-class score",
                         f"{mean_fake_probability:.2%}"),
                        ("Mean real-class score",
                         f"{1 - mean_fake_probability:.2%}"),
                        ("Frames analyzed", str(len(records))),
                        ("XAI method", "Occlusion-based image sensitivity"),
                    ],
                    [
                        "Frame",
                        "Time (s)",
                        "Fake score", "Real score",
                        "Prediction",
                    ],
                    video_frame_rows,
                    (
                        "This report analyzes sampled video frames, not every frame. "
                        "The overall result is based on the mean fake-class probability "
                        "with a 0.5 threshold; it is not a validated video-level classifier. "
                        "Heatmaps show model sensitivity and do not prove manipulation. "
                        "Results are not proof of authenticity."
                        ),
                    images=video_report_images,
                    extra_sections=[
                        (
                            "Temporal Analysis",
                            "The graph shows the model's fake-class probability at each "
                            "sampled time point. Changes across time can help identify "
                            "frames that the model scores differently.",
                        ),
                        (
                            "Explainable AI (XAI)",
                            "The application masks regions of selected face crops and "
                            "measures how the model score changes. The heatmaps are "
                            "approximate explanations of model sensitivity.",
                        ),
                        (
                            "Limitations",
                            "Only a limited number of frames are sampled. Face detection "
                            "can miss faces, and frame-level scores can be affected by "
                            "lighting, image quality, pose, compression, or model bias. "
                            "The results require human review.",
                        ),
                    ],
                )
                st.download_button(
                    label="📄 Download Detailed Video PDF Report",
                    data=video_pdf,
                    file_name="deepfake_video_analysis_report.pdf",
                    mime="application/pdf",
                    key="download_video_pdf",
                )

                # ----------------------------------------------

                st.header("4. Sampled Frames")

                columns = st.columns(4)

                for index, record in enumerate(records):
                    with columns[index % 4]:
                        st.image(
                            record["_preview"],
                            caption=(
                                f"{record['Time (seconds)']} s — "
                                f"{record['Frame prediction']}"
                            ),
                            use_container_width=True,
                        )

            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

        except Exception as error:
            st.error("Video analysis failed.")
            st.exception(error)

else:
    st.warning("Upload a video to begin the analysis.")

st.divider()
st.caption(
    "Project prototype: ViT-based frame classification, temporal "
    "score analysis, and occlusion-based explainability."
)
