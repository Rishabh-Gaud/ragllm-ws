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

agentPrompt = "Task: As a representative of the USC Gould LL.M. Admissions Office your task is to assist students with their queries about the program."
from keybert import KeyBERT

from keybert import KeyBERT
kw_model = KeyBERT(model='intfloat/multilingual-e5-base')

logfile = 'KeyBERT-Log.txt'


def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def extract_keys(text, ngram,dvsity):
    import datetime
    import random
    a = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, ngram), stop_words='english',
                              use_mmr=True, diversity=dvsity, highlight=True)     #highlight=True
    tags = []
    for kw in a:
        tags.append(str(kw[0]))
    timestamped = datetime.datetime.now()
    #LOG THE TEXT AND THE METATAGS
    logging_text = f"LOGGED ON: {str(timestamped)}\nMETADATA: {str(tags)}\nsettings: keyphrase_ngram_range (1,{str(ngram)})  Diversity {str(dvsity)}\n---\nORIGINAL TEXT:\n{text}\n---\n\n"
    writehistory(logging_text)
    return tags

class RagClient:
    def __init__(self):
        
        self.load_dotenv()
        self.client = Groq(
            # organization=os.environ['OPENAI_ORGANIZATION_ID'],
            api_key=os.environ['GROQ_API_KEY'],
        )
        self.website_links = [
            # 'https://gould.usc.edu/academics/degrees/online-llm/',
            # 'https://gould.usc.edu/academics/degrees/llm-1-year/application/',
            # 'https://gould.usc.edu/academics/degrees/two-year-llm/application/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-adr/application/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-ibel/application/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-plcs/application/',
            'https://gould.usc.edu/academics/degrees/online-llm/application/',
            # 'https://gould.usc.edu/academics/degrees/mcl/',
            # 'https://gould.usc.edu/academics/degrees/mcl/application/',
            # 'https://gould.usc.edu/news/what-can-you-do-with-an-llm-degree/',
            # 'https://gould.usc.edu/news/msl-vs-llm-vs-jd-which-law-degree-is-best-for-your-career-path/',
            # 'https://gould.usc.edu/news/three-things-ll-m-grads-wish-they-knew-when-they-started/',
            # 'https://gould.usc.edu/academics/degrees/llm-1-year/',
            # 'https://gould.usc.edu/academics/degrees/two-year-llm/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-adr/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-ibel/',
            # 'https://gould.usc.edu/academics/degrees/llm-in-plcs/'
        ]
        self.dataset =[
            {
            'Title':"Application Instructions - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'Context': """Eligibility
            To qualify for admission to the Master of International Trade Law and Economics, you must have earned your bachelor's degree in any field, within or outside of the United States, by the time you enroll at USC Gould School of Law. Students are required to have a solid foundation in mathematics at the university level, including calculus. If you hold a degree from a university within the United States, USC Graduate Admissions requires that it must be a regionally accredited institution.
            If you are currently earning your degree and will complete it prior to the start of your studies, you may be admitted, but will be required to complete certain continuing registration requirements. You must submit official final transcripts showing your degree was awarded before you may begin your studies.
            We do not require work experience or an LSAT or a GRE score to be considered for admission to any of our graduate degrees. Should you elect to submit an LSAT or GRE score report to supplement your application, such score will not play a key role in our admission decision-making process to ensure an equitable process for all applicants.
            Application Deadlines
            USC Gould School of Law offers two starts in fall and spring. Applications become available in September for Fall semester and in June for Spring semester. To receive priority consideration, apply by our Priority Deadline (see date below for each semester). After our Priority Deadline, applications will only be accepted on a rolling basis as space is available until the final application deadline.
            Below are the start dates and application deadlines:
            Program Start Date
            Priority Deadline
            Final Application Deadline
            Fall 2024: August 26, 2024
            February 1, 2024
            June 1, 2024*
            Spring 2025: January 13, 2025
            September 1, 2024
            November 1, 2024*

            *Final Application Deadline for international students who require initial I-20 documentation for their student visas is May 1, 2024 for Fall semester and October 1, 2024 for Spring semester. We highly encourage you to apply by our Priority Deadline in order to provide sufficient time to process any required documents, including I-20 documentation for student visas, and to apply for USC campus housing.
            If you are planning to take the TOEFL or IELTS exam after our Priority Deadline, we highly recommend that you go ahead and submit your application (and any other completed materials) prior to taking the exam and we will accept your test scores after you receive your official results. The Admissions Committee will make a final admissions decision upon receipt of your complete application file.
            All applicants will be automatically considered for a scholarship award and do not need to submit any further documents to request an award. In addition, applicants who apply by our Priority Deadline will be automatically considered for a housing stipend.
            Application Fee
            Applicants will be charged a $90 application fee.
            Application Process
            To apply for admission, the following information and documents must be submitted:
            1. Online USC Graduate Admission Application
            Note: Once you start your application by clicking above, you will be directed to complete your online application through the USC Office of Graduate Admission application.
            2. Personal Statement
            Following the directions in your online application, provide a two or three-page document that includes your personal, academic, and professional background and your reasons for pursuing the Master of International Trade Law and Economics degree. Please include a description of your quantitative education or experience such as coursework or research projects in math or economics.
            3. Resume/Curriculum Vitae (CV)
            Following the directions in your online application, provide a record of your employment history and list of any distinctions, publications, and licenses/credentials.
            4. Official Transcripts and Degree Verification
            You must have earned your bachelor's degree by the time you enroll at USC. Following the directions provided in your online application, submit official transcripts from all institutions attended.
            Upload scanned copies of your official, up-to-date transcripts into the USC online application system for review for admissions purposes. Please see information on the USC Graduate Admissions website:
            Transcripts Requirements
            FAQ
            Video - Academic History
            Country Requirements
            If you hold any degree from outside of the U.S., USC does not accept or recognize credential evaluation reports from outside agencies; only records issued by your previous institution will be accepted. You may be required to submit English translations of transcripts and diploma(s)/degree certificate(s) - review country requirements above for further information. If you are admitted to USC, you may receive instructions to verify your degree with the International Education Research Foundation (IERF). For detailed information on how to obtain an IERF verification report, please refer to the Degree Progress website.
            5. English Proficiency
            If you are not a U.S. citizen or permanent resident, you must take the Test of English as Foreign Language (TOEFL) or International English Language Testing System (IELTS). TOEFL/IELTS waiver requests will only be considered if (1) your native language is English, or (2) you possess an undergraduate degree, master’s degree, or doctoral degree from an institution in which English is the primary language of instruction in accordance with the USC Graduate Admissions Policy.
            USC Gould does not have a minimum TOEFL or IELTS score; however, we recommend a 90 TOEFL iBT and a 7.0 IELTS or above.
            Please have your official scores sent to USC. Instructions on how to submit scores can be found on the USC Graduate Admissions website.
            Test scores are valid within two years (24 months) of the application date. 
            USC does NOT accept MyBest TOEFL scores of applicants as valid proof of language proficiency. We will consider the single best total TOEFL or IELTS score you submit along with the sub-scores from that test.
            The LSAT or GRE exam is not required for admission. In addition, letters of recommendation are not required.
            The Admissions Committee will not review your application until all transcripts and supporting documents are received. It is your responsibility to ensure that all of your documents are submitted by the final application deadline. Be sure to regularly check the email address you have provided on the application in the event that the Admissions Committee requests additional materials or information.
            Application Status Check
            To ensure you have submitted a complete application, please refer to the USC Graduate Admissions checklists. Our office will contact you via email for any missing or additional requirements. Please contact our office directly at mitle@law.usc.edu with any questions.
            Admission Decisions
            Admissions decisions are made upon receipt of complete application files. Incomplete applications are not reviewed. We encourage you to apply by our Priority Deadline to provide sufficient time for you to complete your application file. Our Admissions Committee will provide admissions decisions generally within 3-4 weeks upon our receipt of completed applications, beginning in January for Fall semester start and beginning in July for Spring semester start. You will be notified via email and postal mail. No admissions decisions will be released over the phone. We will contact you if the Admissions Committee recommends an interview or additional submissions.
            Financial Statement
            If you are admitted, you will be required to provide a financial statement that certifies that you have sufficient funds available to meet your living (housing, meals, etc.) and tuition expenses while at USC (unless you are a U.S. citizen or permanent resident, or have been granted political asylum). At that time, you also must submit a copy of the photo page of your passport (and passport copies of any dependent(s) traveling with you)."""
            },
        ]
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.open_ai_token = os.environ['api_key']
        self.load_embeddings()

        # self.llm_openai = OpenAI(temperature=0.6, openai_api_key=self.open_ai_token)
        
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
            for data in self.dataset:
                splitter = CharacterTextSplitter(chunk_size=330, chunk_overlap=10)
                self.chunks = splitter.split_text(data['Context'])
                # self.chunk_embeddings = [self.model.encode(chunk + str(extract_keys(chunk, 1, 0.35))) for chunk in self.chunks]
                self.chunk_embeddings = []
                self.chunks_with_metadata =[]
                processed_chunks = set()
                for chunk in self.chunks: 
                    tags = extract_keys(chunk, 1, 0.32)
                    # Format the chunk text and its metadata before encoding
                    chunk_with_metadata = f"chunk: {chunk}\n METADATA: {tags} \n "
                    # # Encode the chunk with metadata and append it to self.chunk_embeddings
                    # self.chunk_embeddings.append(self.model.encode(self.chunk_with_metadata))
                    # self.chunks_with_metadata.append(self.chunk_with_metadata)
                    # Check if the chunk has already been processed
                    if chunk_with_metadata not in processed_chunks:
                        self.chunk_embeddings.append(self.model.encode(chunk_with_metadata))
                        self.chunks_with_metadata.append(chunk_with_metadata)
                        processed_chunks.add(chunk_with_metadata)  # Add the processed chunk to the set
                print("chunks_embeddings: ", self.chunk_embeddings)
            with open("chunks_embeddings.pkl", "wb") as f:
                pickle.dump((self.chunks_with_metadata, self.chunk_embeddings), f)

        dimension = self.chunk_embeddings[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(self.chunk_embeddings).astype('float32')
        self.faiss_index.add(embeddings_array)
        # dimension = self.chunk_embeddings[0].shape[1]  # Get the dimension of each embedding vector
        # self.faiss_index = faiss.IndexFlatL2(dimension)  # Initialize the Faiss index
        # embeddings_array = np.concatenate(self.chunk_embeddings, axis=0).astype('float32')  # Concatenate all embeddings into a single array
        # self.faiss_index.add(embeddings_array)  # Add the embeddings to the Faiss index

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
        def concatenate_strings(data_list):
            return "\n".join(map(str, data_list))
        result = concatenate_strings(retrieved_list)
        end_time = time.time()
        print("rag time taken: ",start_time, end_time, end_time - start_time)
        prompt = [{
            "role": "system",
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agentPrompt
        },
            {
            "role": "system",
            "content": "Answer the user's question based on the following information:" +
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
        return result, stream.choices[0].message.content
    def run_chatbot(self):
        st.header("Welcome to LLM Chatbot")

        user_query = st.text_input("Enter the query:", key="user_query")

        if st.button("Answer"):
            if user_query.lower() == "exit":
                st.stop() 
            else:
                output, stream = self.answer(user_query)
                st.write(user_query)
                st.write(stream) 
                st.write(output) 

if __name__ == "__main__":
    chatbot = RagClient()
    chatbot.run_chatbot()