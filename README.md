# AI Voice Meeting Assistant

A local voice-to-text and AI meeting assistant that records microphone audio, transcribes speech in near real time using Whisper, and uses an LLM to generate summaries, extract action items, answer contextual questions, and export structured meeting notes.

This project was built as a lightweight AI engineering demo focused on speech-to-text, LLM integration, transcript analysis, realtime AI workflows, and practical user-facing AI tooling.

---

# Features

- Record audio directly from the microphone
- Stop and resume recording across the same session
- Preserve transcript history across multiple recording sessions
- Display live transcript chunks while recording
- Add visual separators between recording sessions
- Generate AI-powered meeting summaries
- Extract structured action items from meetings
- Ask contextual questions about the transcript using an LLM
- Export meeting transcript, summary, and action items to Markdown
- Save transcript and summary locally
- Store API keys securely using environment variables
- Hybrid local + cloud AI pipeline

---

# Tech Stack

- Python
- CustomTkinter for the desktop GUI
- OpenAI Whisper for local speech-to-text
- Groq API for LLM-powered summarization and transcript Q&A
- SoundDevice for microphone input
- NumPy for audio processing
- python-dotenv for environment variable management
- Markdown export generation

---

# How It Works

The application follows a local + cloud AI workflow:

1. The user starts recording from the desktop GUI.
2. Audio is captured from the microphone in realtime.
3. Whisper transcribes speech locally in small chunks.
4. The transcript appears live inside the application.
5. Recording sessions are preserved within the same conversation context.
6. After recording, the user can:
   - generate summaries
   - extract action items
   - ask contextual questions about the transcript
   - export meeting notes to Markdown
7. The Groq LLM API processes the transcript and returns structured AI responses.

This creates a hybrid AI architecture:
- local speech-to-text inference
- cloud-based LLM reasoning and analysis

---

# Setup

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd STT
```

---

## 2. Create a virtual environment

IMPORTANT: On macOS, use the Python.org installation of Python 3.12 to ensure Tkinter support.

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv venv
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

The application will launch in a desktop GUI window.

---

# Usage

1. Click **Start Recording**
2. Speak into your microphone
3. Watch the transcript appear live
4. Click **Stop Recording**
5. Use:
   - **Create Summary**
   - **Extract Action Items**
   - **Ask AI**
   - **Export Markdown**

---

# Example Questions

```text
What were the main points discussed?
```

```text
What action items were assigned?
```

```text
What deadlines were mentioned?
```

```text
What concerns did the speaker mention?
```

```text
What decisions were made during the meeting?
```

---

# Example Action Items

```text
- Michael will complete the database migration before next Wednesday.
- Daniel will redesign the analytics dashboard by Friday afternoon.
- Christopher will optimize the AI transcription pipeline before Monday’s demo.
```

---

# Project Structure

```text
STT/
├── gui_app.py              # Main desktop application
├── transcribe.py           # Earlier CLI transcription prototype
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # Local API key (not committed)
├── .gitignore              # Git ignored files
├── transcript.txt          # Generated transcript output
├── summary.txt             # Generated AI summary
├── action_items.txt        # Generated action items
├── recording.wav           # Saved recording
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
action_items.txt
```

This prevents:
- API key leaks
- local environment uploads
- large audio file uploads
- generated transcript artifacts from being pushed to GitHub

---

# Demo Focus

This project demonstrates:

- AI-native product thinking
- Speech-to-text integration
- LLM API integration
- Realtime transcript processing
- Local + cloud hybrid AI workflows
- Context-aware transcript analysis
- Transcript-based summarization
- Action-item extraction
- Retrieval-style transcript Q&A
- Practical desktop application development
- Rapid AI prototyping

---

# Future Improvements

Potential next steps:

- Speaker diarization
- Better timestamp support
- Semantic search over meetings
- Vector database integration
- Persistent long-term meeting memory
- Realtime streaming summarization
- Multi-user meeting support
- Deployable web version
- Speaker recognition
- AI-generated meeting titles
- Calendar integrations
- Realtime collaborative meeting notes
- Local LLM support through Ollama
- Vision-aware meeting support

---

# Notes

This project was intentionally designed as a practical AI engineering demo rather than a highly polished commercial product.

The primary focus was:
- rapid prototyping
- practical AI workflows
- realtime interaction
- local AI integration
- modular architecture
- usability
- production-style AI pipeline design

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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Author

Matej Popovski
