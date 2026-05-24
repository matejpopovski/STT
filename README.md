# AI Voice Meeting Assistant

A local voice-to-text and AI meeting assistant that records microphone audio, transcribes speech in near real time using Whisper, and uses an LLM to generate summaries and answer questions about the transcript.

This project was built as a lightweight AI engineering demo focused on speech-to-text, LLM integration, transcript analysis, and practical user-facing AI workflows.

---

# Features

- Record audio directly from the microphone
- Stop and resume recording across the same session
- Display live transcript chunks while recording
- Preserve transcript history across multiple recording sessions
- Add visual separators between recording sessions
- Generate an AI summary of the full transcript
- Ask custom questions about the transcript using an LLM
- Save transcript and summary locally
- Store API keys securely using environment variables

---

# Tech Stack

- Python
- CustomTkinter for the desktop GUI
- OpenAI Whisper for local speech-to-text
- Groq API for LLM-powered summarization and transcript Q&A
- SoundDevice for microphone input
- NumPy for audio processing
- python-dotenv for environment variable management

---

# How It Works

The application follows a simple AI pipeline:

1. The user starts recording from the desktop GUI.
2. Audio is captured from the microphone in small chunks.
3. Whisper transcribes the audio locally.
4. The transcript appears inside the app in near real time.
5. After stopping the recording, the user can:
   - generate a summary
   - ask the AI questions about the transcript
6. The Groq LLM API processes the transcript and returns a structured response.

This creates a local + cloud hybrid AI workflow:
- speech-to-text runs locally
- LLM reasoning runs through the Groq API

---

# Setup

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd STT
```

---

## 2. Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install FFmpeg

Whisper requires FFmpeg.

### macOS

```bash
brew install ffmpeg
```

---

## 5. Create a `.env` file

Create a file named `.env` in the project root:

```bash
touch .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit this file to GitHub.

---

# Running the App

```bash
python gui_app.py
```

The app will open in a desktop window.

---

# Usage

1. Click **Start Recording**
2. Speak into your microphone
3. Watch the transcript appear live
4. Click **Stop Recording**
5. Click **Create Summary** to summarize the transcript
6. Type a question and click **Ask AI** to chat with the transcript

---

# Example Questions

```text
What were the main points discussed?
```

```text
What action items were mentioned?
```

```text
What decisions were made?
```

```text
What concerns did the speaker mention?
```

---

# Project Structure

```text
STT/
├── gui_app.py          # Main desktop application
├── transcribe.py       # Earlier command-line transcription script
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .env                # Local API key, not committed
├── .gitignore          # Files ignored by Git
```

---

# Security Notes

The project uses a `.env` file to store the Groq API key locally.

The `.gitignore` file should include:

```gitignore
.env
venv/
__pycache__/
*.pyc
.DS_Store
*.mp3
*.wav
*.m4a
recording.wav
transcript.txt
summary.txt
```

This prevents API keys, local environments, audio recordings, and generated transcript files from being pushed to GitHub.

---

# Demo Focus

This project demonstrates:

- AI-native product thinking
- Speech-to-text integration
- LLM API integration
- Local + cloud hybrid AI workflow
- Real-time user interaction
- Transcript-based summarization
- Question answering over user-provided context
- Practical desktop application development

---

# Future Improvements

Potential next steps:

- Speaker diarization
- Better timestamp support
- Export summaries to PDF or Markdown
- Semantic search over past meetings
- Local vector database for long-term memory
- Support for multiple LLM providers
- Real-time streaming summarization
- Deployable web version
- Speaker recognition
- Meeting memory persistence across sessions
- AI-generated meeting titles
- Action-item extraction into task lists

---

# Notes

This project was intentionally designed as a practical AI engineering demo rather than a highly polished commercial product.

The focus was:
- rapid prototyping
- practical AI workflows
- local AI integration
- real-time interaction
- usability
- modular architecture

---

# Requirements

Create a `requirements.txt` file with:

```txt
customtkinter
groq
numpy
openai-whisper
python-dotenv
sounddevice
soundfile
torch
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

# Author

Matej Popovski