import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import time
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
import pickle
import streamlit as st
from langchain.llms import OpenAI

class RagClient:
    def __init__(self):
        self.website_links = [
            'https://gould.usc.edu/academics/degrees/online-llm/',
            'https://gould.usc.edu/academics/degrees/llm-1-year/application/',
            'https://gould.usc.edu/academics/degrees/two-year-llm/application/',
            'https://gould.usc.edu/academics/degrees/llm-in-adr/application/',
            'https://gould.usc.edu/academics/degrees/llm-in-ibel/application/',
            'https://gould.usc.edu/academics/degrees/llm-in-plcs/application/',
            'https://gould.usc.edu/academics/degrees/online-llm/application/',
            'https://gould.usc.edu/academics/degrees/mcl/',
            'https://gould.usc.edu/academics/degrees/mcl/application/',
            'https://gould.usc.edu/news/what-can-you-do-with-an-llm-degree/',
            'https://gould.usc.edu/news/msl-vs-llm-vs-jd-which-law-degree-is-best-for-your-career-path/',
            'https://gould.usc.edu/news/three-things-ll-m-grads-wish-they-knew-when-they-started/',
            'https://gould.usc.edu/academics/degrees/llm-1-year/',
            'https://gould.usc.edu/academics/degrees/two-year-llm/',
            'https://gould.usc.edu/academics/degrees/llm-in-adr/',
            'https://gould.usc.edu/academics/degrees/llm-in-ibel/',
            'https://gould.usc.edu/academics/degrees/llm-in-plcs/'
        ]
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.load_dotenv()
        self.open_ai_token = os.environ['api_key']
        self.load_embeddings()

        self.llm_openai = OpenAI(temperature=0.6, openai_api_key=self.open_ai_token)
        
        # if 'listing_history' not in st.session_state:
        #     st.session_state['listing_history'] = []

    def load_dotenv(self):
        load_dotenv()

    def load_embeddings(self):
        if os.path.exists("chunks_embeddings.pkl"):
            with open("chunks_embeddings.pkl", "rb") as f:
                self.chunks, self.chunk_embeddings = pickle.load(f)
        else:
            texts = []
            for link in self.website_links:
                response = requests.get(link)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                texts.append(text)
            combined_text = ",".join(texts)
            splitter = CharacterTextSplitter(chunk_size=1000)
            self.chunks = splitter.split_text(combined_text)
            self.chunk_embeddings = [self.model.encode(chunk) for chunk in self.chunks]

            with open("chunks_embeddings.pkl", "wb") as f:
                pickle.dump((self.chunks, self.chunk_embeddings), f)

        dimension = self.chunk_embeddings[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(self.chunk_embeddings).astype('float32')
        self.faiss_index.add(embeddings_array)

    def answer(self, text):
        # listing_history = st.session_state['listing_history']
        # if len(listing_history) >= 5:
        #     listing_history = listing_history[len(listing_history) - 5:]

        # prompt_template = PromptTemplate(
        #     input_variables=['query_text', 'retrieved', 'listing_history'],
        #     template="Given the following information: '{retrieved}' and {listing_history}. Answer the question: '{query_text}'. "
        # )
        user_content = ""
        for item in text:
        # Check if the role is 'user' and content is not empty
            if item["role"] == "user" and item["content"].strip() != "":
                # Concatenate the user's content
                user_content += item["content"] + " "
            
        start_time = time.time()
        query_embedding = self.model.encode(user_content)
        query_embedding = np.array([query_embedding]).astype('float32')
        k = 10
        D, I = self.faiss_index.search(query_embedding, k)
        retrieved_list = [self.chunks[i] for i in I[0]]
        def concatenate_strings(data_list):
            return "\n".join(map(str, data_list))
        result = concatenate_strings(retrieved_list)
        end_time = time.time()
        # chain = LLMChain(llm=self.llm_openai, prompt=prompt_template)
        # response_stream = chain.run(query_text=text, retrieved=retrieved_list, listing_history=listing_history, stream=True)
        return result, user_content
