import os
import time
from pathlib import Path
import numpy as np
import faiss
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from gtts import gTTS
from pydub import AudioSegment
from textblob import TextBlob
import PyPDF2
import simpleaudio as sa
import spacy
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import logging

# ----------------------------------------------------------------------------
# Setting up logging
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#----------------------------------------------------------------------------
# VoiceTranscriber
#----------------------------------------------------------------------------

class VoiceTranscriber:
    def __init__(self, model = "base"):
        self.model = whisper.load_model(model)
        
    def transcribe(self, audio_path)->str:
        result = self.model.transcribe(str(audio_path))
        text = result.get("text", "").strip()
        logging.info(f"Transcription: {text}")
        return text

#----------------------------------------------------------------------------
# Document Processing and Chunking
#----------------------------------------------------------------------------
class DocumentProcessor:
    def __init__(self, chunk_size = 250, chunk_overlap = 20):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
        
    def process_document(self, document_path: str)->list:
        ext = os.path.splitext(document_path)[1].lower()
        content = ""
        
        if ext == ".pdf":
            logging.info(f"Extracting text from PDF: {document_path}")
            with open(document_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content += page.extract_text() or ""
        else:
            logging.info(f"Reading text file: {document_path}")
            with open(document_path, "r", encoding="utf-8") as file:
                content = file.read()
        
        if not content.strip():
            logging.warning(f"No text extracted from {document_path}")
        
        chunks = self.text_splitter.split_text(content)
        logging.info(f"Document split into {len(chunks)} chunks")
        return chunks
    
    
# ----------------------------------------------------------------------------
# BM25 Retriever for Hybrid Search (Keyword-based)
# ----------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], scores[i]) for i in top_indices]
        return results
    
    
#----------------------------------------------------------------------------
# Embedding and Vector DB creation
#----------------------------------------------------------------------------
class VectorDatabase:
    def __init__(self, embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2", use_hybrid = True):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.embeddings = None
        self.chunk_metadata = [] # to map index back to text
        self.use_hybrid = use_hybrid
        self.bm25_retriever = None
        self.nlp = spacy.load("en_core_web_sm")
        
    def create_index(self, chunks:list):
        logging.info("Creating embeddings for chunks...")
        unique_chunks = list(dict.fromkeys([chunk.lower().strip() for chunk in chunks]))
        logging.info(f"Before deduplication: {len(chunks)} chunks, After: {len(unique_chunks)} chunks") 
        
        embeddings = self.embedding_model.encode(unique_chunks, convert_to_tensor=True)
        self.embeddings = np.array(embeddings).astype("float32")
        dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.add(self.embeddings)
        self.chunk_metadata = unique_chunks
        
        if self.use_hybrid:
            self.bm25_retriever = BM25Retriever(unique_chunks)
        logging.info(f"Index created with {len(unique_chunks)} chunks")
        
    def search(self, query:str,sentiment: dict = None, top_k = 5)->list:
        if self.index is None or len(self.chunk_metadata) == 0:
            logging.warning("Index is empty. Returning empty results.")
            return []
        
        query_vector = np.array([self.embedding_model.encode(query)]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        logging.info("FAISS Top Retrieved Chunks:")
        for idx, i in enumerate(indices[0]):
            logging.info(f"Chunk {i}: {self.chunk_metadata[i]} \nScore: {distances[0][idx]}")
        
        vector_results = []
        seen_texts = set()
        
        sentiment_weights = {
            "joy": 0.9,
            "anger": 1.1,
            "fear": 1.5,
            "sadness": 1.15,
            "neutral": 1.0,
            "surprise": 1.1
        }
        weight = sentiment_weights.get(sentiment["emotion"], 1.0) if sentiment else 1.0
        
        for idx, i in enumerate(indices[0]):
            if i < len(self.chunk_metadata):
                chunk_text = self.chunk_metadata[i]
                if chunk_text not in seen_texts:
                    vector_results.append((chunk_text, distances[0][idx] * weight))
                    seen_texts.add(chunk_text)
        
        # --- BM25 (keyword-based) search ---
        # bm25_results = []
        # if self.use_hybrid and self.bm25_retriever:
        #     bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        #     logging.info("BM25 Top Retrieved Chunks:")
        #     for text, score in bm25_results:
        #         logging.info(f"  - {text[:100]}... (BM25 Score: {score})")
        
        # --- Merge Results ---
        merged_results = {}
        for text, score in vector_results:
            merged_results[text] = merged_results.get(text, 0) + score
        
        # Normalize BM25 scores before inverting
        # max_bm25 = max(bm25_results, key=lambda x: x[1])[1]  # Get max BM25 score
        # for text, bm25_score in bm25_results:
        #     normalized_score = bm25_score / (max_bm25 + 1e-6)  # Scale between 0 and 1
        #     inverted_score = 1 - normalized_score  # Invert while keeping scale meaningful
        #     merged_results[text] = merged_results.get(text, 0) + inverted_score
                    
        sorted_results = sorted(merged_results.items(), key=lambda x: x[1])
        final_results = sorted_results[:top_k]
        logging.info(f"Final Retrieved {len(final_results)} context chunks")
        return final_results
    
#----------------------------------------------------------------------------
# Sentiment Analysis
#----------------------------------------------------------------------------
class SentimentAnalyzer:
    def __init__(self, model = "j-hartmann/emotion-english-distilroberta-base"):
        self.emotion_pipeline = pipeline("text-classification", model=model)
    
    def analyze(self, text:str)->dict:
        result = self.emotion_pipeline(text)[0]
        sentiment = {
            "emotion" : result["label"],
            "score" : result["score"]
        }
        logging.info(f"Detected emotion: {sentiment}")
        return sentiment
    
#----------------------------------------------------------------------------
# LLM Context Enrichment
#----------------------------------------------------------------------------
class LLMResponder:
    def __init__(self, model = "meta-llama/Llama-3.2-1B-Instruct"):
        self.pipeline = pipeline("text-generation", model = model, device_map = "auto")
        
    def generate_response(self, query:str, contexts:list, sentiment:dict)->str:
        context_text = "\n".join([f"- {text[:250]}..." for text, _ in contexts])  # Limit chunk size
        tone_mapping = {
            "joy" : "warm and encouraging",
            "anger" : "calm and understanding",
            "fear" : "reassuring and supportive",
            "sadness" : "compassionate and comforting",
            "neutral" : "clear and informative",
            "surprise" : "measured and thoughtful"
        }
        tone = tone_mapping.get(sentiment["emotion"], "neutral")
        query = f"{query}\n\n[IMPORTANT]: Make sure your response is {tone}. Use appropriate language and structure to match this emotion. Ensure your response **does not ask questions or offer choices**."

        messages = [
            {
                "role": "system",
                "content": f"""
                "You are medical/financial/technical assistant. Your responses must be without asking questions or offering options to the user."
                Your responses must be:
                - **Clinically accurate** or factually correct based on the provided context.
                - **Highly Emotionally aligned** with the user's detected sentiment **{sentiment['emotion']}**
                - **Concise and complete**, with a response limited to 5-10 sentences.
                
                
                The user's detected emotion is **{sentiment['emotion']}**, which means your response should be **{tone}**.

                Here is relevant context being provided to you which is highly important:
                {context_text}

                Now, respond to the following patient query:
                """
            },
            {"role": "user", "content": query},
        ]
        
        logging.info("Generating response...")
        response = self.pipeline(messages, max_new_tokens = 300, do_sample = True)
        
        try:
            # Adjust extraction based on your model’s output structure
            answer = response[0]['generated_text'][-1]['content']
        except Exception as e:
            answer = response[0]['generated_text']
        
        logging.info(f"Response: {answer}")
        return answer
    

#----------------------------------------------------------------------------
# Text To Speech
#----------------------------------------------------------------------------

class TextToSpeech:
    def __init__(self, lang = "en"):
        self.lang = lang
        
    def speak(self, text:str):
        logging.info(f"Speaking: {text}")
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)

        # Paths for audio files
        mp3_path = os.path.join(temp_dir, "output.mp3")
        wav_path = os.path.join(temp_dir, "output.wav")

        # Convert text to speech and save as MP3
        tts = gTTS(text=text, lang=self.lang)
        tts.save(mp3_path)

        # Convert MP3 to WAV for simpleaudio
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

        # Play the WAV file
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        # Optional cleanup
        os.remove(mp3_path)
        os.remove(wav_path)

        logging.info("Speech complete")

#----------------------------------------------------------------------------
# Main Program
#----------------------------------------------------------------------------

class VoiceActivatedRAGSystem:
    def __init__(self, document_folder: str):
        self.transcriber = VoiceTranscriber(model="base")
        self.doc_processor = DocumentProcessor(chunk_overlap=20, chunk_size=250)
        self.vector_db = VectorDatabase(use_hybrid = True)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.llm_responder = LLMResponder()
        self.tts = TextToSpeech(lang="en")
        self._load_documents(document_folder)
        
    def _load_documents(self, document_foldeer: str):
        all_chunks = []
        doc_path = Path(document_folder)
        for filename in doc_path.iterdir():
            if filename.suffix in {".txt", ".pdf"}:
                chunks = self.doc_processor.process_document(str(filename))
                all_chunks.extend(chunks)
        
        self.vector_db.create_index(all_chunks)
        
    def run(self, audio_input_path:str):
        start_time = time.time()
        query = self.transcriber.transcribe(audio_input_path)
           
        sentiment = self.sentiment_analyzer.analyze(query)
        contexts = self.vector_db.search(query, sentiment=sentiment, top_k=5)
        logging.info(f"Retrieved Context Chunks: {contexts}")
        with ThreadPoolExecutor() as executor:
            future_response = executor.submit(self.llm_responder.generate_response, query, contexts, sentiment)
            response = future_response.result()
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.2f} seconds")
            executor.submit(self.tts.speak, response)
        

if __name__== "__main__":
    
    script_dir = Path(__file__).parent
    
    # Path to the folder containing documents and audio_file
    document_folder = script_dir / "docs"
    audio_path = script_dir / "scenario1(exampleMentioned).mp3"
    
    # Allow user to provide custom paths
    custom_doc_path = input(f"Enter document folder path (press Enter to use default: {document_folder}): ").strip()
    custom_audio_path = input(f"Enter audio file path (press Enter to use default: {audio_path}): ").strip()

    if custom_doc_path:
        document_folder = Path(custom_doc_path)
    if custom_audio_path:
        audio_path = Path(custom_audio_path)

    if not document_folder.exists():
        logging.error(f"Error: Document folder '{document_folder}' does not exist!")
        exit(1)
    if not audio_path.exists():
        logging.error(f"Error: Audio file '{audio_path}' does not exist!")
        exit(1)
        
    # Creating an instance of the system
    rag_system = VoiceActivatedRAGSystem(document_folder=document_folder)
    rag_system.run(audio_path)
        