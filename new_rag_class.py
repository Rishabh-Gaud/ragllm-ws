# import os
# import pickle
# import fitz
# import numpy as np
# import faiss
# import time
# from langchain import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.llms import OpenAI
# from langchain_text_splitters import CharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv

# from groq import Groq

# load_dotenv()

# agentPrompt = "Task: As a representative of the USC Gould LL.M. Admissions Office your task is to assist students with their queries about the program."

# class PdfRagQueryProcessor:
#     def __init__(self):
#         self.pdf_path = "example.pdf"
#         self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
#         self.api_key = os.environ['api_key']
#         self.client = Groq(
#             # organization=os.environ['OPENAI_ORGANIZATION_ID'],
#             api_key=os.environ['GROQ_API_KEY'],
#         )
#         self.text_splitter = CharacterTextSplitter(
#             separator="\n\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             is_separator_regex=False,
#         )
#         self.model = SentenceTransformer(self.model_name)
#         # self.llm_openai = OpenAI(temperature=0.3, openai_api_key=self.api_key)
#         self.load_or_compute_embeddings()
#         self.init_faiss_index()

#     def load_or_compute_embeddings(self):
#         if os.path.exists("chunks_embeddings.pkl"):
#             with open("chunks_embeddings.pkl", "rb") as f:
#                 self.chunks, self.chunk_embeddings = pickle.load(f)
#         else:
#             with fitz.open(self.pdf_path) as doc:
#                 text = ""
#                 for page in doc:
#                     text += page.get_text()
#                 text = text.replace(" ", "")
#                 self.chunks = self.text_splitter.split_text(text)
#                 self.chunk_embeddings = [self.model.encode(chunk) for chunk in self.chunks]
#                 with open("chunks_embeddings.pkl", "wb") as f:
#                     pickle.dump((self.chunks, self.chunk_embeddings), f)

#     def init_faiss_index(self):
#         dimension = self.chunk_embeddings[0].shape[0]
#         self.faiss_index = faiss.IndexFlatL2(dimension)
#         embeddings_array = np.array(self.chunk_embeddings).astype('float32')
#         self.faiss_index.add(embeddings_array)

#     # def answer(self, text, listing_history):
#     def answer(self, text):
#         # if len(listing_history) >= 5:
#         #     listing_history = listing_history[-5:]
        
#         user_content = ""
#         # for item in text:
#         # # Check if the role is 'user' and content is not empty
#         #     if item["role"] == "user" and item["content"].strip() != "":
#         #         # Concatenate the user's content
#         #         user_content += item["content"] + " "

#         # prompt_template = PromptTemplate(
#         #     input_variables=['query_text', 'retrieved', 'listing_history'],
#         #     template="""Based on the user's query "{query_text}" and the provided relevant information: {retrieved}, your task is to synthesize a clear, accurate, and concise response. Follow these steps:

#         #     1. Analyze the provided information to understand its relevance to the user's query.
#         #     2. Construct a response that directly addresses the user's query using the information given. Aim for clarity and brevity.
#         #     3. If the information does not fully address the user's query, clearly state that the query cannot be completely answered with the available information and suggest the need for additional sources or clarification.
#         #     4. you can also use the data in {listing_history} for  answering
#         #     Remember, your goal is to provide a helpful and precise answer that is directly based on the provided information. Avoid making assumptions beyond the given content."""
#         # )
#         start_time = time.time()
#         query_embedding = self.model.encode(text)
#         query_embedding = np.array([query_embedding]).astype('float32')
#         k = 10
#         D, I = self.faiss_index.search(query_embedding, k)
#         retrieved_list = [self.chunks[i] for i in I[0]]
#         middle_time = time.time()
#         print("rag time taken: ", start_time, middle_time, middle_time - start_time)
#         prompt = [{
#             "role": "system",
#             "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
#           agentPrompt
#             },
#             {
#             "role": "system",
#             "content": "Answer the user's question based on the following information and use metaData as well to give answer: " +
#           str(retrieved_list)
#             },
#             {
#             "role": "user",
#             "content": "Answer the user's question with very consice and short" +
#           text
#             },
            
#         ]
        
        
#         stream = self.client.chat.completions.create(
#             model="mixtral-8x7b-32768", 
#             messages=prompt,
#         )
#         end_time = time.time()
#         print("llm time time taken: ",middle_time, end_time, end_time - middle_time)
#         # chain = LLMChain(llm=self.llm_openai, prompt=prompt_template)
#         # response_stream = chain.run(query_text=text, retrieved=retrieved_list, listing_history=listing_history, stream=True)
#         return stream.choices[0].message.content, middle_time - start_time, end_time - middle_time

#     # def interact(self):
#     #     listing_history = []
#     #     while True:
#     #         user_query = input("Enter the query here: ")
#     #         if user_query == "exit":
#     #             break
#     #         else:
#     #             output, time_consumed = self.answer(user_query)
#     #             print(output)
#     #             listing_history.append(output)

# # Usage:
# # pdf_processor = PDFQueryProcessor(pdf_path="example.pdf", model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=os.environ['api_key'])
# # pdf_processor.interact()
