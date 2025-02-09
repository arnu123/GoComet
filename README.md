# Voice-Activated RAG System


## 🚀 Features

- 🎤 Voice Transcription: Converts speech into text using OpenAI Whisper.

- 📄 Document Processing: Ingests text/PDF documents for efficient retrieval.

- 😃 Sentiment-Aware Responses: Adapts response tone based on detected sentiment.

- 🔊 Text-to-Speech (TTS): Converts responses into natural-sounding audio output.

## 📥 Installation

- 1️⃣ Clone the Repository

```
git clone https://github.com/arnu123/GoComet.git
cd GoComet
```
- 2️⃣ Install Dependencies
* Ensure Python 3.8+ is installed. Then, run:

```
pip install -r requirements.txt
```
- 3️⃣ Upgrade Transformers (Recommended)
```
pip install --upgrade transformers
```

## Spacy Model Installation
- Download the English language model:
```bash
python -m spacy download en_core_web_sm
```
## 🎬 Usage

### Running the System

- To launch GoComet, simply run:
```
python gocomet.py
```
- Mention the path to the docs and audio_file when prompted. To use the ones already in the directory, just press enter when prompted.

## Workflow

The system transcribes the provided audio file.

It analyzes the sentiment of the transcribed text.

Relevant context is retrieved from stored documents.

The language model generates a contextual response.

The response is converted to speech and played back.



