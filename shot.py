import gradio as gr
import whisper
import os
import subprocess
from pathlib import Path
import tempfile

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
LANGUAGES = ["auto"] + list(whisper.tokenizer.LANGUAGES.keys())

def extract_audio(video_path):
    """Extract audio from video file using ffmpeg"""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.close()
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", temp_audio.name
    ]
    subprocess.run(cmd, capture_output=True)
    return temp_audio.name

def transcribe_file(file_path, model_name, language):
    """Transcribe a single file"""
    model = whisper.load_model(model_name)
    
    # Check if video file
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if Path(file_path).suffix.lower() in video_exts:
        audio_path = extract_audio(file_path)
    else:
        audio_path = file_path
    
    # Transcribe
    lang = None if language == "auto" else language
    result = model.transcribe(audio_path, language=lang)
    
    # Cleanup temp audio if created
    if audio_path != file_path:
        os.unlink(audio_path)
    
    return result["text"]

def process_files(files, model_name, language):
    """Process multiple files"""
    if not files:
        return "No files uploaded", None
    
    results = []
    for file in files:
        try:
            text = transcribe_file(file.name, model_name, language)
            results.append(f"=== {Path(file.name).name} ===\n{text}\n")
        except Exception as e:
            results.append(f"=== {Path(file.name).name} ===\nError: {str(e)}\n")
    
    full_text = "\n".join(results)
    
    # Save to temp file for download
    temp_output = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt", encoding='utf-8')
    temp_output.write(full_text)
    temp_output.close()
    
    return full_text, temp_output.name

def process_folder(folder_path, model_name, language):
    """Process all audio/video files in folder"""
    if not folder_path:
        return "No folder selected", None
    
    audio_exts = ['.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a', '.wma']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    all_exts = audio_exts + video_exts
    
    files = []
    for ext in all_exts:
        files.extend(Path(folder_path).glob(f"*{ext}"))
        files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not files:
        return "No audio/video files found in folder", None
    
    results = []
    for file_path in files:
        try:
            text = transcribe_file(str(file_path), model_name, language)
            results.append(f"=== {file_path.name} ===\n{text}\n")
        except Exception as e:
            results.append(f"=== {file_path.name} ===\nError: {str(e)}\n")
    
    full_text = "\n".join(results)
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt", encoding='utf-8')
    temp_output.write(full_text)
    temp_output.close()
    
    return full_text, temp_output.name

with gr.Blocks() as demo:
    gr.Markdown("# Whisper Transcription")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=WHISPER_MODELS, value="base", label="Model")
        language_dropdown = gr.Dropdown(choices=LANGUAGES, value="auto", label="Language")
    
    with gr.Tab("Files"):
        file_input = gr.File(file_count="multiple", label="Upload Audio/Video Files")
        file_button = gr.Button("Transcribe Files")
        file_output = gr.Textbox(label="Transcription", lines=10)
        file_download = gr.File(label="Download Transcription")
        
        file_button.click(
            process_files,
            inputs=[file_input, model_dropdown, language_dropdown],
            outputs=[file_output, file_download]
        )
    
    with gr.Tab("Folder"):
        folder_input = gr.Textbox(label="Folder Path")
        folder_button = gr.Button("Transcribe Folder")
        folder_output = gr.Textbox(label="Transcription", lines=10)
        folder_download = gr.File(label="Download Transcription")
        
        folder_button.click(
            process_folder,
            inputs=[folder_input, model_dropdown, language_dropdown],
            outputs=[folder_output, folder_download]
        )

if __name__ == "__main__":
    demo.launch(share=True)
