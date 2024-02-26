import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
import pickle
website_links = [
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
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
load_dotenv()
open_ai_token = os.environ['api_key']
if os.path.exists("chunks_embeddings.pkl"):
    with open("chunks_embeddings.pkl", "rb") as f:
        chunks, chunk_embeddings = pickle.load(f)
else:
    texts = []
    for link in website_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    combined_text = ",".join(texts)
    splitter = CharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_text(combined_text)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    chunk_embeddings = [model.encode(chunk) for chunk in chunks]

    with open("chunks_embeddings.pkl", "wb") as f:
        pickle.dump((chunks, chunk_embeddings), f)

dimension = chunk_embeddings[0].shape[0]
faiss_index = faiss.IndexFlatL2(dimension)
embeddings_array = np.array(chunk_embeddings).astype('float32')
faiss_index.add(embeddings_array)
from langchain.llms import OpenAI
llm_openai = OpenAI(temperature=0.6,openai_api_key=open_ai_token)

beginSentence = "Hey there, I'm your personal AI student ambassador, how can I help you?"
agentPrompt = "Task: As a professional university student ambassador, your responsibilities are comprehensive and details. You establish a positive and trusting rapport with student and solve him query"

listing_history=[]

class LlmClient:
    def __init__(self):
        self.client = OpenAI(
            # organization=os.environ['OPENAI_ORGANIZATION_ID'],
            api_key=os.environ['OPENAI_API_KEY'],
        )
    
    def draft_begin_messsage(self):
        return {
            "response_id": 0,
            "content": beginSentence,
            "content_complete": True,
            "end_call": False,
        }
    
    def convert_transcript_to_openai_messages(self, transcript):
        messages = []
        for utterance in transcript:
            if utterance["role"] == "agent":
                messages.append({
                    "role": "assistant",
                    "content": utterance['content']
                })
            else:
                messages.append({
                    "role": "user",
                    "content": utterance['content']
                })
        return messages

    def prepare_prompt(self, request):
        prompt = [{
            "role": "system",
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agentPrompt
        }]
        transcript_messages = self.convert_transcript_to_openai_messages(request['transcript'])
        for message in transcript_messages:
            prompt.append(message)

        if request['interaction_type'] == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "(Now the user has not responded in a while, you would say:)",
            })
        return prompt

    def draft_response(self, request): 
        preparedPrompt = self.prepare_prompt(request)
        # if len(listing_history)>=5:
        #     listing_history=listing_history[len(listing_history)-5:]

        prompt_template = PromptTemplate(
            input_variables=['query_text', 'retrieved','listing_history'],
            template="Given the following information: '{retrieved}' and  . Answer the question: '{query_text}'. "
        )
        query_embedding = model.encode(preparedPrompt) #preparedPrompt text
        query_embedding = np.array([query_embedding]).astype('float32')
        k = 10
        D, I = faiss_index.search(query_embedding, k)
        retrieved_list = [chunks[i] for i in I[0]]
        chain = LLMChain(llm=llm_openai, prompt=prompt_template)
        stream = chain.run(query_text=preparedPrompt, retrieved=retrieved_list,listing_history="listing_history",stream=True)            
        # stream = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=preparedPrompt,
        #     stream=True,
        # )
        print("hello yaha pe code likha h>>>>>>>>>>>>>>>>>>>>>>>>>>", stream)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {
                    "response_id": request['response_id'],
                    "content": chunk.choices[0].delta.content,
                    "content_complete": False,
                    "end_call": False,
                }
        
        yield {
            "response_id": request['response_id'],
            "content": "",
            "content_complete": True,
            "end_call": False,
        }
