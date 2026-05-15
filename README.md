# Speaksure
This project is a Python-based speech processing and audio analysis application that allows users to upload or record audio files and convert speech into text while generating multiple visual analytics.
import speech_recognition as sr
from gtts import gTTS
import os
import uuid
import gradio as gr
import threading
import time
import matplotlib.pyplot as plt
from collections import Counter
import string
import librosa
import librosa.display
import numpy as np
from textblob import TextBlob
from pydub import AudioSegment
import pandas as pd  # For displaying history in a table
from pydub.utils import which
import subprocess

history = []  # Global list to store history of inputs & transcriptions

if which("ffmpeg") is None:
    # Set the path to where you installed FFmpeg
    os.environ["PATH"] += os.pathsep + r"C:\FFmpeg"  # Adjust this path to match your FFmpeg installation

def convert_to_wav(audio_path):
    """
    Converts an audio file to WAV format if it's not already in WAV.
    Ensures the audio is mono and has a sampling rate of 16kHz.
    Returns the path to the WAV file.
    """
    try:
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        # Special handling for opus files
        if file_ext == ".opus":
            # Try using subprocess to convert opus directly to wav using ffmpeg
            wav_filename = f"converted_{uuid.uuid4().hex}.wav"
            subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-ar', '16000',  # Set sample rate to 16kHz
                '-ac', '1',      # Convert to mono
                '-y',            # Overwrite output file if it exists
                wav_filename
            ], check=True, capture_output=True)
            print(f"Converted opus file to {wav_filename}")
            return wav_filename
            
        # Handle other audio formats
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        if file_ext != ".wav":
            wav_filename = f"converted_{uuid.uuid4().hex}.wav"
            audio.export(wav_filename, format="wav")
            print(f"Converted audio saved to {wav_filename}")
            return wav_filename
        else:
            return audio_path
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return None

def speech_to_text(audio):
    recognizer = sr.Recognizer()
    if isinstance(audio, str):
        audio_file = audio
    else:
        audio_file = audio.name

    wav_file = convert_to_wav(audio_file)
    if not wav_file:
        return "Sorry, there was an error processing the audio file."

    try:
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio_data)
        print("Transcription: " + text)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return "Sorry, could not understand the audio."
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return f"Could not request results; {e}"
    finally:
        if wav_file != audio_file and os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"Deleted temporary WAV file: {wav_file}")

def text_to_speech(text):
    if not text or text.startswith("Sorry") or text.startswith("Could not"):
        return None

    try:
        tts = gTTS(text=text, lang='en')
        filename = f"temp_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        print(f"Generated speech saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

def generate_word_frequency_chart(text):
    if not text or text.startswith("Sorry") or text.startswith("Could not"):
        return None

    try:
        translator = str.maketrans('', '', string.punctuation)
        words = text.translate(translator).lower().split()

        word_counts = Counter(words)
        most_common = word_counts.most_common(10)
        if not most_common:
            return None

        words, counts = zip(*most_common)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(words, counts, color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Word Frequencies')
        plt.xticks(rotation=45)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                     ha='center', va='bottom')

        plot_filename = f"plot_{uuid.uuid4().hex}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"Word frequency chart saved to {plot_filename}")
        return plot_filename
    except Exception as e:
        print(f"Error generating word frequency chart: {e}")
        return None

def generate_waveform_chart(audio_path):
    try:
        y, sr_ = librosa.load(audio_path, sr=None)
        time_axis = np.linspace(0, len(y) / sr_, num=len(y))

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, y, color='skyblue')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        plt.tight_layout()

        waveform_filename = f"waveform_{uuid.uuid4().hex}.png"
        plt.savefig(waveform_filename)
        plt.close()
        print(f"Waveform chart saved to {waveform_filename}")
        return waveform_filename
    except Exception as e:
        print(f"Error generating waveform chart: {e}")
        return None

def generate_sentiment_chart(text):
    if not text or text.startswith("Sorry") or text.startswith("Could not"):
        return None

    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            sentiment_label = 'Positive'
        elif sentiment < 0:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        sizes = [0, 0, 0]
        if sentiment > 0:
            sizes[0] = sentiment
        elif sentiment < 0:
            sizes[1] = abs(sentiment)
        else:
            sizes[2] = 1

        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['green', 'red', 'gray']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title('Sentiment Analysis')
        plt.axis('equal')

        sentiment_filename = f"sentiment_{uuid.uuid4().hex}.png"
        plt.tight_layout()
        plt.savefig(sentiment_filename)
        plt.close()
        print(f"Sentiment chart saved to {sentiment_filename}")
        return sentiment_filename
    except Exception as e:
        print(f"Error generating sentiment chart: {e}")
        return None

def process_audio(audio):
    """
    Main function called on 'Submit'. Returns transcribed_text, TTS audio,
    and chart paths if successful.
    """
    transcribed_text = speech_to_text(audio)
    speech_file = text_to_speech(transcribed_text)
    chart_file = generate_word_frequency_chart(transcribed_text)

    # Determine final path for waveform chart
    if isinstance(audio, str):
        audio_path = audio
    else:
        audio_path = audio.name
    waveform_file = generate_waveform_chart(audio_path)

    sentiment_file = generate_sentiment_chart(transcribed_text)

    # Store in history if it's a valid transcription
    if transcribed_text and not transcribed_text.startswith("Sorry"):
        # You can store the actual audio path or just the name
        history.append({
            "Audio_File": os.path.basename(audio_path),
            "Transcription": transcribed_text
        })

    # Schedule deletion of generated files
    if speech_file and os.path.exists(speech_file):
        def delete_file(path, delay=10):
            time.sleep(delay)
            try:
                os.remove(path)
                print(f"Deleted temporary file: {path}")
            except Exception as e:
                print(f"Error deleting file {path}: {e}")
        threading.Thread(target=delete_file, args=(speech_file,)).start()

    if chart_file and os.path.exists(chart_file):
        def delete_chart(path, delay=60):
            time.sleep(delay)
            try:
                os.remove(path)
                print(f"Deleted temporary chart file: {path}")
            except Exception as e:
                print(f"Error deleting chart file {path}: {e}")
        threading.Thread(target=delete_chart, args=(chart_file,)).start()

    if waveform_file and os.path.exists(waveform_file):
        def delete_waveform(path, delay=60):
            time.sleep(delay)
            try:
                os.remove(path)
                print(f"Deleted temporary waveform file: {path}")
            except Exception as e:
                print(f"Error deleting waveform file {path}: {e}")
        threading.Thread(target=delete_waveform, args=(waveform_file,)).start()

    if sentiment_file and os.path.exists(sentiment_file):
        def delete_sentiment(path, delay=60):
            time.sleep(delay)
            try:
                os.remove(path)
                print(f"Deleted temporary sentiment file: {path}")
            except Exception as e:
                print(f"Error deleting sentiment file {path}: {e}")
        threading.Thread(target=delete_sentiment, args=(sentiment_file,)).start()

    return transcribed_text, speech_file, chart_file, waveform_file, sentiment_file

def get_history():
    """
    Return a DataFrame containing all past audio files and their transcriptions.
    """
    if len(history) == 0:
        return pd.DataFrame(columns=["Audio_File", "Transcription"])
    return pd.DataFrame(history)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    # --- Top Section: Logo + Title side by side ---
    with gr.Row():
        gr.Markdown("Stuttering Visual Analytics. Upload or Record your Audio file (mp3, wav, opus) to Convert it to Text.")

    # --- Create Tabs ---
    with gr.Tabs():
        # --- 1) Main Tab with input/output ---
        with gr.Tab("Main"):
            audio_input = gr.Audio(type="filepath", label="Record Stuttering Audio Here")
            submit_btn = gr.Button("Submit")

            text_output = gr.Textbox(label="Transcribed Text")
            speech_output = gr.Audio(label="Generated Speech")

            with gr.Accordion("Click to View Stuttering Audio Charts", open=False):
                chart_output = gr.Image(label="Word Frequency Chart")
                waveform_output = gr.Image(label="Audio Waveform")
                sentiment_output = gr.Image(label="Sentiment Analysis Chart")

            # On submit, process audio
            submit_event = submit_btn.click(
                fn=process_audio,
                inputs=audio_input,
                outputs=[
                    text_output,         # Transcribed text
                    speech_output,       # Generated speech
                    chart_output,        # Word frequency chart
                    waveform_output,     # Waveform chart
                    sentiment_output     # Sentiment analysis chart
                ]
            )

        # --- 2) History Tab ---
        with gr.Tab("History"):
            gr.Markdown("All previously processed audio and their transcriptions:")
            # We'll create an empty DataFrame component. We store it in a variable.
            history_table = gr.Dataframe(
                value=pd.DataFrame(columns=["Audio_File", "Transcription"]),
                headers=["Audio_File", "Transcription"],
                datatype=["str", "str"],
                interactive=False,
                label="History"
            )

    # --- After audio is processed, update the History table ---
    submit_event.then(
        fn=get_history,
        inputs=None,
        outputs=history_table
    )

# Launch the app
app, local_url, share_url = demo.launch(share=True)

# Optionally print the URLs
# print("Local URL:", local_url)
# print("Public (Share) URL:", share_url)
