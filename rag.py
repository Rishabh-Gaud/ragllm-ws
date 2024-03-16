import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import time
import faiss
import numpy as np
from groq import Groq
from langchain.prompts import PromptTemplate
import pickle
import streamlit as st
from langchain.llms import OpenAI
from tqdm.rich import trange, tqdm
from rich import console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
# console = Console(width=110)
from transformers import pipeline
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

agentPrompt = "Task: As a representative of the USC Gould LL.M. Admissions Office your task is to assist students with their queries about the program."

class RagClient:
    def __init__(self):
        
        self.load_dotenv()
        
        self.chunks_with_metadata =[]
        self.client = Groq(
            # organization=os.environ['OPENAI_ORGANIZATION_ID'],
            api_key=os.environ['GROQ_API_KEY'],
        )
        self.dataset =[
            
            {
            'Title':"Career Opportunities - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career Opportunities - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - Alternative Dispute Resolution Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Alternative Dispute Resolution Certificate.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in ADR.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum - Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of Laws (LLM) - Online.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Curriculum Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"LLM - 1 year",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/LLM - 1 year.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of Comparative Law (MCL) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Comparative Law (MCL) Degree.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws (LLM) - Online.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of Laws (LLM) in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws (LLM) in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of Laws in Alternative Dispute Resolution (LLM in ADR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws in Alternative Dispute Resolution (LLM in ADR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Master of Science in Innovation Economics, Law and Regulation (MIELR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Science in Innovation Economics, Law and Regulation (MIELR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid - Alternative Dispute Resolution Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Alternative Dispute Resolution Certificate.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - LLM in ADR.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition & Financial Aid LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Tuition and Financial Aid Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition and Financial Aid Master of Laws (LLM) - Online.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Two-Year Extended Master of Laws (LLM)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Two-Year Extended Master of Laws (LLM).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Types of Aid - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Types of Aid - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"USC Gould -  Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould -  Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"USC Gould - Centre for Dispute Resolution_",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Centre for Dispute Resolution_.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"USC Gould - Housing Flyer",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Housing Flyer.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"USC Gould - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Alternative Dispute Resolution (ADR) Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Alternative Dispute Resolution (ADR) Certificate.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions - Alternate Dispute Resolution (ADR) Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Alternate Dispute Resolution (ADR) Certificate.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions Extended Master of Laws (LLM)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Extended Master of Laws (LLM).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in ADR.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Laws (LLM) - Online.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Application Instructions Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Laws (LLM) Degree.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Bar and Certificate Tracks Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Bar and Certificate Tracks Master of Laws (LLM) - Online.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in ADR.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Campus Housing - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career and Bar - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career and Bar - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in ADR.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career and Bar - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in International Business and Economic Law.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career and Bar - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career and Bar Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar Master of Comparative Law (MCL).txt',
            "tags" : ["ucla"]
            },
            {
            'Title':"Career Opportunities - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career Opportunities - Master of Dispute Resolution (MDR).txt',
            "tags" : ["ucla"]
            },
        ]
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        # self.open_ai_token = os.environ['api_key']
        self.load_embeddings()



    def load_dotenv(self):
        load_dotenv()



    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    def load_embeddings(self):
        chunks_from_file=[]
        if os.path.exists("./chunks_text.txt"):
            with open("./chunks_text.txt", "rb") as f:
                text= pickle.load(f)
        else:
            texts = []
            for data in self.dataset:
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
                with open(data['filename'], encoding="utf8") as f:
                    fulltext = f.read()
                chunks = splitter.split_text(fulltext)
                for chunk in chunks:
                    chunks_from_file.append(chunk)
            complete_text="".join(chunks_from_file)
            with open("./chunks_text.txt",'wb') as f:
                pickle.dump(complete_text,f)
        dimension = self.chunk_embeddings[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(self.chunk_embeddings).astype('float32')
        self.faiss_index.add(embeddings_array)                         
        

    def answer(self, text):
        user_content = ""
        start_time = time.time()
        # for item in text:
        # # Check if the role is 'user' and content is not empty
        #     if item["role"] == "user" and item["content"].strip() != "":
        #         # Concatenate the user's content
        #         user_content += item["content"] + " "
            
        query_embedding = self.model.encode(text)
        query_embedding = np.array([query_embedding]).astype('float32')
        k = 10
        D, I = self.faiss_index.search(query_embedding, k)
        retrieved_list = [self.chunks[i] for i in I[0]]
        retrieved_object_list = []
        for i in I[0]: 

            retrieved_object_list.append({
                    "content": self.chunks[i]
                })
                
        
        def concatenate_strings(data_list):
            return "\n".join(map(str, data_list))
        result = concatenate_strings(retrieved_list)
        end_time = time.time()
        print("rag time taken: ",start_time, end_time, end_time - start_time)
        prompt = [{
            "role": "system",
            "content": '##Objective/nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agentPrompt
        },
            {
            "role": "system",
            "content": "Answer the user's question based on the following information and use metaData as well to give answer: " +
          result
        },
            {
            "role": "user",
            "content": "Answer the user's question with very consice and short" +
          text
        },
            
            ]
        stream = self.client.chat.completions.create(
            model="mixtral-8x7b-32768", 
            messages=prompt,
        )
        # chain = LLMChain(llm=self.llm_openai, prompt=prompt_template)
        # response_stream = chain.run(query_text=text, retrieved=retrieved_list, listing_history=listing_history, stream=True)
        return retrieved_object_list, stream.choices[0].message.content

    






























