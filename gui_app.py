import os
import queue
import threading
import tempfile
import wave
from datetime import datetime

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
        self.geometry("1400x1150")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.recording = False
        self.audio_chunks = []
        self.audio_queue = queue.Queue()
        self.transcript = ""
        self.summary = ""
        self.action_items = ""

        self.title_label = ctk.CTkLabel(
            self,
            text="AI Voice Meeting Assistant",
            font=("Arial", 40, "bold")
        )
        self.title_label.pack(pady=20)

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=("Arial", 18)
        )
        self.status_label.pack(pady=5)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=15)

        self.record_button = ctk.CTkButton(
            button_frame,
            text="Start Recording",
            command=self.start_recording,
            width=200,
            height=50,
            font=("Arial", 17)
        )
        self.record_button.grid(row=0, column=0, padx=8)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            width=200,
            height=50,
            font=("Arial", 17),
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=8)

        self.summary_button = ctk.CTkButton(
            button_frame,
            text="Create Summary",
            command=self.create_summary,
            width=200,
            height=50,
            font=("Arial", 17),
            state="disabled"
        )
        self.summary_button.grid(row=0, column=2, padx=8)

        self.action_button = ctk.CTkButton(
            button_frame,
            text="Extract Action Items",
            command=self.extract_action_items,
            width=220,
            height=50,
            font=("Arial", 17),
            state="disabled"
        )
        self.action_button.grid(row=0, column=3, padx=8)

        self.ask_button = ctk.CTkButton(
            button_frame,
            text="Ask AI",
            command=self.ask_ai,
            width=180,
            height=50,
            font=("Arial", 17),
            state="disabled"
        )
        self.ask_button.grid(row=0, column=4, padx=8)

        self.export_button = ctk.CTkButton(
            button_frame,
            text="Export Markdown",
            command=self.export_markdown,
            width=200,
            height=50,
            font=("Arial", 17),
            state="disabled"
        )
        self.export_button.grid(row=0, column=5, padx=8)

        transcript_label = ctk.CTkLabel(
            self,
            text="Live Transcript",
            font=("Arial", 24, "bold")
        )
        transcript_label.pack(pady=(20, 10))

        self.transcript_box = ctk.CTkTextbox(
            self,
            width=1250,
            height=230,
            font=("Arial", 18)
        )
        self.transcript_box.pack(pady=10)
        self.transcript_box.insert("end", "Live transcript will appear here...\n")

        summary_label = ctk.CTkLabel(
            self,
            text="AI Summary",
            font=("Arial", 24, "bold")
        )
        summary_label.pack(pady=(15, 10))

        self.summary_box = ctk.CTkTextbox(
            self,
            width=1250,
            height=150,
            font=("Arial", 18)
        )
        self.summary_box.pack(pady=10)
        self.summary_box.insert("end", "Summary will appear here...\n")

        action_label = ctk.CTkLabel(
            self,
            text="Action Items",
            font=("Arial", 24, "bold")
        )
        action_label.pack(pady=(15, 10))

        self.action_box = ctk.CTkTextbox(
            self,
            width=1250,
            height=130,
            font=("Arial", 18)
        )
        self.action_box.pack(pady=10)
        self.action_box.insert("end", "Action items will appear here...\n")

        question_label = ctk.CTkLabel(
            self,
            text="Ask AI About The Transcript",
            font=("Arial", 24, "bold")
        )
        question_label.pack(pady=(15, 10))

        self.question_entry = ctk.CTkEntry(
            self,
            width=1250,
            height=50,
            font=("Arial", 18),
            placeholder_text="Example: What were the main concerns discussed?"
        )
        self.question_entry.pack(pady=10)

        answer_label = ctk.CTkLabel(
            self,
            text="AI Response",
            font=("Arial", 24, "bold")
        )
        answer_label.pack(pady=(15, 10))

        self.answer_box = ctk.CTkTextbox(
            self,
            width=1250,
            height=150,
            font=("Arial", 18)
        )
        self.answer_box.pack(pady=10)
        self.answer_box.insert("end", "AI response will appear here...\n")

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())
            self.audio_chunks.append(indata.copy())

    def start_recording(self):
        self.recording = True

        separator = "\n\n================ NEW RECORDING ================\n\n"

        if self.transcript.strip():
            self.transcript += separator
            self.transcript_box.insert("end", separator)

        self.audio_chunks = []

        self.summary_box.delete("1.0", "end")
        self.action_box.delete("1.0", "end")
        self.answer_box.delete("1.0", "end")
        self.question_entry.delete(0, "end")

        self.status_label.configure(text="Recording...")

        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.summary_button.configure(state="disabled")
        self.action_button.configure(state="disabled")
        self.ask_button.configure(state="disabled")
        self.export_button.configure(state="disabled")

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback
        )

        self.stream.start()

        threading.Thread(
            target=self.live_transcribe_loop,
            daemon=True
        ).start()

    def stop_recording(self):
        self.recording = False

        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        self.status_label.configure(text="Recording stopped")

        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.summary_button.configure(state="normal")
        self.action_button.configure(state="normal")
        self.ask_button.configure(state="normal")
        self.export_button.configure(state="normal")

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

                        self.transcript_box.insert(
                            "end",
                            text.strip() + " "
                        )

                        self.transcript_box.see("end")

            except queue.Empty:
                continue

    def transcribe_audio_array(self, audio_data):
        audio_data = audio_data.flatten().astype(np.float32)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
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

        audio_data = np.concatenate(
            self.audio_chunks,
            axis=0
        ).flatten()

        with wave.open("recording.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        with open("transcript.txt", "w") as f:
            f.write(self.transcript)

    def create_summary(self):
        if not self.transcript.strip():
            return

        self.status_label.configure(text="Generating summary...")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI meeting assistant.

Create a concise professional summary with:
- Main discussion points
- Important details
- Decisions made
- Action items mentioned
- Follow-up questions if relevant

Only use information from the transcript.
"""
                },
                {
                    "role": "user",
                    "content": self.transcript
                }
            ],
            temperature=0.2,
        )

        self.summary = response.choices[0].message.content

        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", self.summary)

        with open("summary.txt", "w") as f:
            f.write(self.summary)

        self.status_label.configure(text="Summary created")

    def extract_action_items(self):
        if not self.transcript.strip():
            return

        self.status_label.configure(text="Extracting action items...")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI meeting assistant.

Extract clear action items from the transcript.

Rules:
- Use bullet points.
- Include owner names if mentioned.
- Include deadlines if mentioned.
- If no action items are mentioned, say: "No clear action items were mentioned."
- Do not invent tasks that are not in the transcript.
"""
                },
                {
                    "role": "user",
                    "content": self.transcript
                }
            ],
            temperature=0.1,
        )

        self.action_items = response.choices[0].message.content

        self.action_box.delete("1.0", "end")
        self.action_box.insert("end", self.action_items)

        with open("action_items.txt", "w") as f:
            f.write(self.action_items)

        self.status_label.configure(text="Action items extracted")

    def ask_ai(self):
        question = self.question_entry.get()

        if not question.strip():
            return

        self.status_label.configure(text="Generating AI response...")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI assistant analyzing a meeting transcript.

Answer the user's question using ONLY the transcript.
If the answer is not mentioned, clearly say so.
Be concise, accurate, and helpful.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Transcript:
{self.transcript}

Question:
{question}
"""
                }
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        self.answer_box.delete("1.0", "end")
        self.answer_box.insert("end", answer)

        self.status_label.configure(text="AI response created")

    def export_markdown(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"meeting_export_{timestamp}.md"

        markdown_content = f"""# AI Meeting Assistant Export

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Transcript

{self.transcript if self.transcript.strip() else "No transcript available."}

---

## Summary

{self.summary if self.summary.strip() else "No summary generated yet."}

---

## Action Items

{self.action_items if self.action_items.strip() else "No action items extracted yet."}
"""

        with open(filename, "w") as f:
            f.write(markdown_content)

        self.status_label.configure(text=f"Exported to {filename}")


if __name__ == "__main__":
    app = VoiceApp()
    app.mainloop()