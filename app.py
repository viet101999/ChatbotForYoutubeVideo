import os
import re
import subprocess
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import List

import PIL.Image
import faiss
import google.generativeai as genai
import numpy as np
import whisper
from fastapi import FastAPI, UploadFile
from sentence_transformers import SentenceTransformer

from module.crawl_youtube import (
    download_video,
    download_and_convert_video,
    get_video_title
)


# Define the ChatBot class
class ChatBot:
    def __init__(self):
        self.app = FastAPI()  # Define FastAPI instance here
        self.GOOGLE_API_KEY = ""
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.transcription_model = whisper.load_model("base")  # Whisper model for transcription
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Text embedding model

        # FAISS index for semantic search
        self.dimension = 384  # Embedding dimension for the MiniLM model
        self.index = faiss.IndexFlatL2(self.dimension)

        self.save_folder = "downloads"

        # Store metadata
        self.video_metadata = []

        self.current_link = ""
        self.video_title = ""
        self.current_video_title = ""
        self.current_video_path = ""
        self.current_video_file = ""

        # API Endpoints
        self.define_routes()  # Define API endpoints

    def define_routes(self):
        @self.app.post("/upload/")
        async def upload_video(file: UploadFile):
            file_path = f"./{file.filename}"

            video_title = Path(file_path).stem
            self.video_title = video_title
            cleaned_video_title = re.sub(r'[^a-zA-Z0-9]', '', video_title).lower()

            video_path = os.path.join("downloads", f"{cleaned_video_title}.mp4")
            audio_path = os.path.join("downloads", f"{cleaned_video_title}.wav")
            transcript_path = os.path.join("downloads", f"{cleaned_video_title}.txt")

            if not os.path.exists(video_path):
                shutil.copy(file_path, video_path) 

            if not os.path.exists(audio_path):
                shutil.copy(file_path, audio_path)

            transcript_path = os.path.join("downloads", f"{video_title}.txt")
            if os.path.exists(transcript_path):
                self.load_transcript(transcript_path)
            else:
                transcripts = self.transcribe_audio(audio_path)
                self.write_transcript(transcript_path, transcripts)

            # Build FAISS index
            self.build_index()

            if self.is_video_exist(video_name=cleaned_video_title):
                pass
            else:
                self.upload_video(video_path)
            
            self.current_video_title = cleaned_video_title
            self.current_video_path = video_path
            self.current_video_file = genai.get_file(f"files/{cleaned_video_title}")

            return {"name": video_title}

        @self.app.post("/process_youtube/")
        async def process_youtube(link: str):
            if link == self.current_link:
                video_title = self.video_title
                cleaned_video_title = re.sub(r'[^a-zA-Z0-9]', '', video_title).lower()
            else:
                video_title = get_video_title(link)
                cleaned_video_title = re.sub(r'[^a-zA-Z0-9]', '', video_title).lower()

                self.current_link = link
                self.video_title = video_title

            video_path = os.path.join("downloads", f"{cleaned_video_title}.mp4")
            audio_path = os.path.join("downloads", f"{cleaned_video_title}.wav")
            transcript_path = os.path.join("downloads", f"{cleaned_video_title}.txt")

            if not os.path.exists(video_path):
                video_path = download_video(link, self.save_folder)

            if not os.path.exists(audio_path):
                audio_path = download_and_convert_video(link, self.save_folder)

            if os.path.exists(transcript_path):
                self.load_transcript(transcript_path)
            else:
                transcripts = self.transcribe_audio(audio_path)
                self.write_transcript(transcript_path, transcripts)

            # Build FAISS index
            self.build_index()

            if self.is_video_exist(video_name=cleaned_video_title):
                pass
            else:
                self.upload_video(video_path)

            self.current_video_title = cleaned_video_title
            self.current_video_path = video_path
            self.current_video_file = genai.get_file(f"files/{cleaned_video_title}")

            return {"name": video_title}

        @self.app.get("/query_chatbot_search/")
        async def query_chatbot_search(question: str):
            result = self.answer_query(question)
            return result

        @self.app.get("/query_chatbot_search_full/")
        async def query_chatbot_search_full(question: str):
            result = self.get_full_context(question)
            return result

        @self.app.get("/query_chatbot_text/")
        async def query_chatbot_text(question: str):
            transcript = " ".join([f"[{entry['start']} - {entry['end']}] {entry['text']}" for entry in self.video_metadata])
            content = "Below is a transcript of a video/audio\n{transcript}\n\nplease answer the user's question as accurately and concisely as possible\nQuestion: {query}"
            content = content.replace("{transcript}", transcript).replace("{query}", question)

            response = self.model.generate_content(
                contents=content
            )
            result = response.text
            return result

        @self.app.get("/query_chatbot_image/")
        async def query_chatbot_image(question: str, image_file: UploadFile):
            if self.is_video_exist(video_name=self.current_video_title):
                video_file = self.current_video_file
            else:
                video_file = self.upload_video(self.current_video_path)

            # Read the uploaded file as bytes
            image_data = await image_file.read()
            # Use BytesIO to create a file-like object for PIL
            image = PIL.Image.open(BytesIO(image_data))

            prompt = """
            Watch each frame in the video carefully and answer the user's question as accurately and concisely as possible.
            Only base your answers strictly on what information is available in the video attached.
            Do not make up any information that is not part of the video and do not be to verbose, be to the point.

            Questions: {query}
            """

            prompt = prompt.replace("{query}", question)
            response = self.model.generate_content(
                [
                    video_file,
                    image,
                    prompt
                ]
            )
            result = response.text
            return result

    # Helper methods for audio processing, transcription, and search
    def extract_audio(self, video_path: str, audio_path: str):
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
        ])

    def transcribe_audio(self, audio_path: str):
        result = self.transcription_model.transcribe(audio_path)
        return result['segments']  # Transcripts with timestamps

    def build_index(self):
        embeddings = []
        # metadata = []
        for segment in self.video_metadata:
            text = segment['text']
            embedding = self.embedding_model.encode(text)
            embeddings.append(embedding)
            # metadata.append(segment)

        faiss_embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(faiss_embeddings)
        # self.video_metadata.extend(metadata)

    def write_transcript(self, transcript_path: str, transcript: list):
        with open(transcript_path, 'w') as w:
            for segment in transcript:
                w.write(f"[{segment['start']} - {segment['end']}] {segment['text']}\n")

    def load_transcript(self, transcript_path: str):
        with open(transcript_path, 'r') as r:
            data_all = r.readlines()
            self.video_metadata = []
            metadata = []
            for data in data_all:
                start = data.split("]  ")[0].split(" - ")[0].replace("[", "").strip()
                end = data.split("]  ")[0].split(" - ")[-1].replace("]", "").strip()
                text = data.split("]  ")[-1].strip()
                segment = {
                    "start": start,
                    "end": end,
                    "text": text
                }
                metadata.append(segment)

            self.video_metadata.extend(metadata)

    def upload_video(self, video_path: str):
        # Upload the video
        name_video = Path(video_path).stem

        print(f"Uploading file...")
        video_file = genai.upload_file(
            path=video_path,
            name=f"files/{name_video}",
            display_name=name_video
        )
        print(f"Completed upload: {video_file.uri}")

        # Check whether the file is ready to be used.
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        return video_file

    def is_video_exist(self, video_name: str):
        name = f"files/{video_name}"
        list_file = []
        for f in genai.list_files():
            list_file.append(f.name)
        if name in list_file:
            print("File was uploaded")
            return True
        else:
            return False

    def answer_query(self, query: str):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), 1)

        if distances[0][0] < 1.0:  # Distance threshold
            metadata = self.video_metadata[indices[0][0]]
            return {
                "text": metadata['text'],
                "start_time": metadata['start'],
                "end_time": metadata['end']
            }
        else:
            return {"message": "No relevant content found"}

    def get_full_context(self, query: str):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), 5)

        if distances[0][0] < 1.0:
            context = ""
            for idx in indices[0]:
                segment = self.video_metadata[idx]
                context += f"{segment['text']} "
            return {"answer": context.strip()}
        else:
            return {"message": "No relevant content found"}


# Instantiate ChatBot and expose the app
chatbot = ChatBot()
app = chatbot.app