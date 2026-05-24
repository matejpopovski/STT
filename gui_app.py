import os
import queue
import threading
import tempfile
import wave
from pathlib import Path

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = whisper.load_model("base")


class VoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Voice Meeting Assistant")
        self.geometry("900x700")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.recording = False
        self.audio_chunks = []
        self.audio_queue = queue.Queue()
        self.transcript = ""

        self.title_label = ctk.CTkLabel(
            self,
            text="AI Voice Meeting Assistant",
            font=("Arial", 30, "bold")
        )
        self.title_label.pack(pady=20)

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=("Arial", 16)
        )
        self.status_label.pack(pady=5)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=15)

        self.record_button = ctk.CTkButton(
            button_frame,
            text="Start Recording",
            command=self.start_recording,
            width=180,
            height=45
        )
        self.record_button.grid(row=0, column=0, padx=10)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            width=180,
            height=45,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=10)

        self.summary_button = ctk.CTkButton(
            button_frame,
            text="Create Summary",
            command=self.create_summary,
            width=180,
            height=45,
            state="disabled"
        )
        self.summary_button.grid(row=0, column=2, padx=10)

        self.transcript_box = ctk.CTkTextbox(
            self,
            width=820,
            height=300,
            font=("Arial", 15)
        )
        self.transcript_box.pack(pady=15)
        self.transcript_box.insert("end", "Live transcript will appear here...\n")

        self.summary_box = ctk.CTkTextbox(
            self,
            width=820,
            height=200,
            font=("Arial", 15)
        )
        self.summary_box.pack(pady=15)
        self.summary_box.insert("end", "Summary will appear here after recording...\n")

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())
            self.audio_chunks.append(indata.copy())

    def start_recording(self):
        self.recording = True
        self.audio_chunks = []
        self.transcript = ""

        self.transcript_box.delete("1.0", "end")
        self.summary_box.delete("1.0", "end")

        self.status_label.configure(text="Recording...")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.summary_button.configure(state="disabled")

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

        threading.Thread(target=self.live_transcribe_loop, daemon=True).start()

    def stop_recording(self):
        self.recording = False

        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        self.status_label.configure(text="Recording stopped")
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.summary_button.configure(state="normal")

        self.save_full_audio()

    def live_transcribe_loop(self):
        buffer = []

        while self.recording:
            try:
                chunk = self.audio_queue.get(timeout=1)
                buffer.append(chunk)

                duration = sum(len(x) for x in buffer) / SAMPLE_RATE

                if duration >= CHUNK_SECONDS:
                    audio_data = np.concatenate(buffer, axis=0)
                    buffer = []

                    text = self.transcribe_audio_array(audio_data)

                    if text.strip():
                        self.transcript += " " + text.strip()
                        self.transcript_box.insert("end", text.strip() + " ")
                        self.transcript_box.see("end")

            except queue.Empty:
                continue

    def transcribe_audio_array(self, audio_data):
        audio_data = audio_data.flatten().astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        result = model.transcribe(temp_path)
        os.remove(temp_path)

        return result["text"]

    def save_full_audio(self):
        if not self.audio_chunks:
            return

        audio_data = np.concatenate(self.audio_chunks, axis=0).flatten()

        with wave.open("recording.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        with open("transcript.txt", "w") as f:
            f.write(self.transcript)

    def create_summary(self):
        if not self.transcript.strip():
            self.summary_box.delete("1.0", "end")
            self.summary_box.insert("end", "No transcript available.")
            return

        self.status_label.configure(text="Generating summary...")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI meeting assistant.

Create a clean summary with:
1. Main points
2. Important details
3. Decisions made
4. Action items
5. Follow-up questions
"""
                },
                {
                    "role": "user",
                    "content": self.transcript
                }
            ],
            temperature=0.2,
        )

        summary = response.choices[0].message.content

        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", summary)

        with open("summary.txt", "w") as f:
            f.write(summary)

        self.status_label.configure(text="Summary created")


if __name__ == "__main__":
    app = VoiceApp()
    app.mainloop()
