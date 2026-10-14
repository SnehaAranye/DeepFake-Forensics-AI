
import os
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
st.title("🔍 DeepFake Forensics AI")
st.subheader(
    "Vision Transformer | Temporal Analysis | Explainable AI"
)

st.write(
    "Analyze video frames, compare model scores over time, "
    "and inspect an approximate visual explanation."
)

st.info(
    "Research prototype: results depend on the trained model and "
    "the quality of the input. Predictions are not proof of manipulation."
)


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
                st.warning("The AI model flagged this image as potentially manipulated.")
            else:
                st.success("The AI model classified this image as likely authentic.")
            st.write("### 📊 Model Score Comparison")
            st.progress(float(fake_score), text=f"Fake class: {fake_score:.2%}")
            st.progress(float(real_score), text=f"Real class: {real_score:.2%}")
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


        except Exception as error:
            st.error("Image analysis failed.")
            st.exception(error)

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

            with st.spinner("Analyzing audio with Wav2Vec2..."):
                processor, model, audio_labels = load_audio_model()

                probabilities = predict_audio_file(
                    audio_path, processor, model
                )

            ranking = np.argsort(probabilities)[::-1]

            st.subheader("Audio classification scores")

            score_data = [
                {
                    "Model label": audio_labels[int(index)],
                    "Score": float(probabilities[index]),
                }
                for index in ranking
            ]

            st.dataframe(
                pd.DataFrame(score_data),
                use_container_width=True,
                hide_index=True,
            )

            top_index = int(ranking[0])
            top_label = audio_labels[top_index]

            st.metric(
                "Highest-scoring class",
                top_label,
                f"{float(probabilities[top_index]):.1%}",
            )

            # Only interpret the result when class names are explicit.
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

            if fake_indices and real_indices:
                fake_score = sum(
                    float(probabilities[i]) for i in fake_indices
                )
                real_score = sum(
                    float(probabilities[i]) for i in real_indices
                )

                st.write(
                    "Interpretation:",
                    "Fake-like audio" if fake_score > real_score
                    else "Real-like audio",
                )
            else:
                st.warning(
                    "The model's class labels could not be mapped "
                    "reliably to real and fake. Review the raw labels "
                    "above rather than assuming which one means fake."
                )

            st.caption(
                "Audio model results are experimental and depend on "
                "the model's training data. Only the first 20 seconds "
                "are analyzed. Results are not proof of authenticity."
            )
            st.divider()
            st.subheader("🧠 Audio Explainable AI")
            if st.button("Generate Audio Explanation", key="audio_xai"):
                try:
                    with st.spinner("Analyzing audio segments..."):
                        waveform, sr, sensitivity_df, predicted_class, base_score = (
                            explain_audio_segments(
                                audio_path,
                                feature_extractor,
                                model,
                                labels
                                )
                            )
                        duration = len(waveform) / sr
                        times = np.arange(len(waveform)) / sr
                        waveform_df = pd.DataFrame({
                            "Time (seconds)": times,
                            "Amplitude": waveform
                        })
                        st.write(f"**Model's highest-scoring class:** {predicted_class}")
                        st.write(f"**Score for that class:** {base_score:.2%}")
                        st.write("### Audio waveform")
                        st.line_chart(
                            waveform_df,
                            x="Time (seconds)",
                            y="Amplitude"
                        )
                        st.write("### Which segments affect the score?")
                        st.bar_chart(
                            sensitivity_df,
                            x="Start (seconds)",
                            y="Score change"
                        )
                        st.dataframe(sensitivity_df, use_container_width=True)
                        with st.expander("How to understand this explanation"):
                            st.write(
                                "Each bar shows how much the model's selected-class score "
                                "changed when that audio segment was replaced with silence."
                            )
                            st.write(
                                "A larger change means greater sensitivity to that segment. "
                                "It does not prove that the voice is fake."
                            )
                except Exception as error:
                    st.error("Could not generate the audio explanation.")
                    st.exception(error)


        except Exception as error:
                    st.error("Could not generate the audio explanation.")
                    st.exception(error)
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    st.warning(
                        "The temporary audio file could not be removed. "
                        "Close any process using it and try again."
                    )


# -------------------- VIDEO ANALYSIS --------------------

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

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download frame analysis CSV",
                    data=csv_data,
                    file_name="deepfake_frame_analysis.csv",
                    mime="text/csv",
                )

                st.header("3. Explainable AI (XAI)")

                # Explain the frame with the highest fake-class score.
                selected_record = max(
                    records,
                    key=lambda record: record["Fake probability"],
                )

                selected_image = selected_record["_image"]
                selected_probabilities = selected_record["_probabilities"]
                target_index = int(np.argmax(selected_probabilities))

                st.write(
                    f"Explanation frame: {selected_record['Frame']} "
                    f"at {selected_record['Time (seconds)']} seconds. "
                    f"The selected class is **{labels[target_index]}**."
                )

                xai_col1, xai_col2 = st.columns(2)

                with xai_col1:
                    st.image(
                        selected_image,
                        caption="Original sampled frame",
                        use_container_width=True,
                    )

                with st.spinner("Generating the occlusion explanation..."):
                    explanation = create_occlusion_explanation(
                        selected_image,
                        processor,
                        model,
                        target_index,
                    )

                with xai_col2:
                    st.image(
                        explanation,
                        caption="Occlusion-based explanation overlay",
                        use_container_width=True,
                    )

                st.caption(
                    "Interpretation: warmer regions represent patches whose "
                    "occlusion changed the selected class probability more "
                    "relative to the displayed color scale. This coarse "
                    "explanation can be noisy and is not a segmentation mask "
                    "or proof of a manipulated region."
                )

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
