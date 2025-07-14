import os
import logging
import shutil
import whisper  # type: ignore
import ffmpeg  # type: ignore
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from groq import Groq  # type: ignore
import cv2  # type: ignore
from dotenv import load_dotenv

load_dotenv()

# --- Create Necessary Folders ---
os.makedirs("logs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("audio", exist_ok=True)

# --- Logging Setup: File + Terminal ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("logs/app.log")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Flask & Config ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY is not set. Please configure it in your environment.")
MODEL_NAME = "llama3-8b-8192"

# --- Load Image Captioning ---
logger.info("Loading ViT-GPT2 captioning model...")
model_name_captioning = "nlpconnect/vit-gpt2-image-captioning"
captioning_model = VisionEncoderDecoderModel.from_pretrained(model_name_captioning)
feature_extractor = ViTImageProcessor.from_pretrained(model_name_captioning)
tokenizer = AutoTokenizer.from_pretrained(model_name_captioning)
device = "cuda" if torch.cuda.is_available() else "cpu"
captioning_model.to(device)

# --- Load Whisper ---
logger.info("Loading Whisper speech-to-text model...")
whisper_model = whisper.load_model("base")


# --- Utility Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def extract_frames(video_path, output_folder, interval=0.5):
#     video_filename = os.path.basename(video_path)
#     frames_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0] + "_frames")
#     os.makedirs(frames_subfolder, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return [], "Could not open video."

#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     frame_id = 0
#     extracted = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if int(cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) % int(frame_rate * interval * 1000) < (1000 / frame_rate):
#             path = os.path.join(frames_subfolder, f"frame_{frame_id:04d}.jpg")
#             cv2.imwrite(path, frame)
#             extracted.append(path)
#             frame_id += 1

#     cap.release()
#     return extracted, None

def extract_frames(video_path, output_folder, interval=2.5):
    video_filename = os.path.basename(video_path)
    frames_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0] + "_frames")
    os.makedirs(frames_subfolder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Could not open video."

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate

    extracted = []
    timestamp = 0
    frame_id = 0

    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(frames_subfolder, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(path, frame)
        extracted.append(path)
        frame_id += 1
        timestamp += interval

    cap.release()
    return extracted, None


# def generate_caption(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
#         output_ids = captioning_model.generate(pixel_values, attention_mask=attention_mask, max_length=50, do_sample=True, top_k=50, top_p=0.95)
#         return tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     except Exception as e:
#         logger.error(f"Caption error: {e}")
#         return f"Error: {e}"

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        pixel_values = inputs.pixel_values
        attention_mask = inputs.get("attention_mask", None)

        output_ids = captioning_model.generate(
            pixel_values,
            attention_mask=attention_mask,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Caption error: {e}")
        return f"Error: {e}"

def extract_audio(video_path, audio_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run(overwrite_output=True, quiet=True)
        )
        logger.info(f"Audio extracted to {audio_path}")
        return True
    except ffmpeg.Error as e:
        logger.error(f"Audio extraction failed: {e.stderr.decode()}")
        return False


def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        logger.info(f"Transcription completed")
        return result['text']
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


def prepare_caption_text(captions_list):
    return "\n".join(f"Frame {i + 1}: {c.replace('Caption:', '').strip()}" for i, c in enumerate(captions_list))


# def summarize_combined(captions_text, transcript, style="general"):
#     client = Groq(api_key=GROQ_API_KEY)

#     # system_prompt = {
#     #     "general": "You summarize video content of crime scenes with visuals and audio dialogue.",
#     #     "law_enforcement": "You write reports as a law enforcement officer summarizing incidents involving crime and arrests."
#     #     }
#     if style == "general":
#         system_prompt = "You summarize video content of crime scenes with visuals and audio dialogue."
#     elif style == "law_enforcement":
#         system_prompt = "You write reports as a law enforcement officer summarizing incidents involving crime and arrests."
#     else:
#         system_prompt = "You summarize video content."
#     prompt = (
#         f"Frame-by-frame visual descriptions:\n{captions_text}\n\n"
#         f"Audio Transcript:\n{transcript}\n\n"
#         "Write a detailed narrative summary involving all actions, speech, and key moments. "
#         "Base your summary ONLY on the visual and audio inputs. "
#         "Do NOT add any facts or assumptions that are not clearly present. "
#         "If information is unclear or missing, avoid making up details."
#     )
#     try:
#         response = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=400,
#             temperature=0.4
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Groq error: {e}")
#         return "Groq summarization failed."

def summarize_combined(captions_text, transcript, style="general"):
    client = Groq(api_key=GROQ_API_KEY)

    if style == "general":
        system_prompt = (
            "You are a visual and audio summarizer trained to interpret surveillance, dashcam, or bodycam videos. "
            "Summarize all notable movements, interactions, and events based strictly on the captions and transcript."
        )
    elif style == "law_enforcement":
        system_prompt = (
            "You are a law enforcement officer writing a bodycam report. "
            "Describe in a professional and structured manner the sequence of events including visual scenes and audio conversations. "
            "Mention subject behavior (e.g., fleeing, resisting, compliance), vehicle status (e.g., parked, moving), officer actions, and any escalation. "
            "Use neutral tone, avoid assumptions, and only include what is evident from the captions and audio."
        )
    else:
        system_prompt = "You summarize video content."

    prompt = (
        f"Frame-by-frame visual descriptions (as seen through a bodycam or dashcam):\n{captions_text}\n\n"
        f"Audio Transcript:\n{transcript}\n\n"
        "Generate a detailed, factual summary that reflects the incident as a professional observer. "
        "Include actions, vehicle status, officer attempts, and escalation. "
        "Use short paragraphs to convey time-based progression. Do not speculate."
        "Base your summary ONLY on the visual and audio inputs. "
        "Do NOT add any facts or assumptions that are not clearly present. "
        "If information is unclear or missing, avoid making up details."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Groq summarization failed."

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'videoFile' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['videoFile']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    logger.info(f"Received file: {filename}")

    # Extract frames
    frames, err = extract_frames(video_path, app.config['UPLOAD_FOLDER'])
    if err:
        return jsonify({"error": err}), 500

    logger.info(f"Extracted {len(frames)} frames at intervals of 2.5 seconds")

    # Caption generation
    captions = [generate_caption(fp) for fp in sorted(frames)]
    caption_text = prepare_caption_text(captions)

    # Audio + transcription
    audio_path = f"audio/{os.path.splitext(filename)[0]}.wav"
    if extract_audio(video_path, audio_path):
        transcript = transcribe_audio(audio_path)
    else:
        transcript = ""

    # Final summaries
    general_summary = summarize_combined(caption_text, transcript, "general")
    law_summary = summarize_combined(caption_text, transcript, "law_enforcement")

    # Cleanup
    try:
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + "_frames"))
        os.remove(video_path)
        os.remove(audio_path)
        logger.info("Cleanup completed")
    except Exception as cleanup_err:
        logger.warning(f"Cleanup failed: {cleanup_err}")

    return jsonify({
        "captions": captions,
        "transcript": transcript,
        "general_summary": general_summary,
        "law_enforcement_summary": law_summary
    })


# --- Start App ---
if __name__ == '__main__':
    print("App running at: http://127.0.0.1:8009")
    app.run(debug=True, host='0.0.0.0', port=8009, threaded=False)
