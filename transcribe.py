import sys
import os
from pathlib import Path

import whisper
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def summarize_text(transcript: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You summarize transcripts clearly and concisely."
            },
            {
                "role": "user",
                "content": f"Summarize this transcript:\n\n{transcript}"
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py audio_file.m4a")
        return

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        return

    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(str(audio_path))

    transcript = result["text"]

    print("\nTRANSCRIPT:\n")
    print(transcript)

    print("\nGenerating summary with Groq...\n")

    summary = summarize_text(transcript)

    with open("transcript.txt", "w") as f:
        f.write(transcript)

    with open("summary.txt", "w") as f:
        f.write(summary)
        
    print("SUMMARY:\n")
    print(summary)


if __name__ == "__main__":
    main()