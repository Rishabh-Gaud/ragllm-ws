import asyncio
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import json
import os
import time
from dotenv import load_dotenv
class PineconeDocumentProcessor:
    def __init__(self):
        
        load_dotenv()
        
        # Initialize SentenceTransformer model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize Pinecone API
        self.pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        
        # Initialize Pinecone index
        self.index_name = "rag-pipeline"
        self.index = self.pc.Index(self.index_name)
        
        # Initialize OpenAIEmbeddings
        # self.embeddings = OpenAIEmbeddings()
        
    def process_documents(self, dataset):
        for file in dataset:
            filename = file["filename"]
            loader = TextLoader(filename)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            docs = text_splitter.split_documents(documents)
            for i in range(len(docs)):
                vec = self.model.encode(docs[i].page_content)
                metadata = {
                    'title': file['Title'],
                    'data': docs[i].page_content,
                    'filename': file['filename'],
                    'url': file['URL'],
                    'program': file['program']
                }
                id = file['filename'] + str(i)
                upsert_response = self.index.upsert(vectors=[
                    {
                        "id": id,
                        "values": vec,
                        "metadata": metadata
                    }
                ])
                print("done...")
                
    def query_index(self, query, top_k=5, filter_filename=None):
        query_vector = self.model.encode(query).tolist()
        start= time.time()
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            # include_values=True,
            include_metadata=True,
            filter={"program": {"$eq": filter_filename}} if filter_filename else None
        )
        end = time.time()
        print(end-start)
        # print(result)
        modifiedData = ""
        for doc in  result.matches:
            modifiedData  += json.dumps(doc.metadata['data'])
        return modifiedData


    def query_index1(self, query, top_k=10, program=None):
        user_content = ""
        for item in query:
        # Check if the role is 'user' and content is not empty
            if item["role"] == "user" and item["content"].strip() != "":
                # Concatenate the user's content
                user_content += item["content"] + " "
           
        query_vector = self.model.encode(user_content).tolist()
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            # include_values=True,
            include_metadata=True,
            filter={"program": {"$eq": program}} if program else None
        )
        # print(result)
        modifiedData = ""
        for doc in  result.matches:
            modifiedData  += json.dumps(doc.metadata['data'])
        return modifiedData
# Instantiate the PineconeDocumentProcessor class
# processor = PineconeDocumentProcessor()

# dataset =[        
#             {
#             'Title':"Career Opportunities - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career Opportunities - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Curriculum - Alternative Dispute Resolution Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Alternative Dispute Resolution Certificate.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Curriculum - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in ADR.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Curriculum - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Curriculum - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"Curriculum - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Curriculum - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Curriculum - Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of Laws (LLM) - Online.txt',
#             "tags" : "usc",
#             "program": "Online LLM"
#             },
#             {
#             'Title':"Curriculum Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"LLM - 1 year",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/LLM - 1 year.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Master of Comparative Law (MCL) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Comparative Law (MCL) Degree.txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws (LLM) - Online.txt',
#             "tags" : "usc",
#             "program": "Online LLM"
#             },
#             {
#             'Title':"Master of Laws (LLM) in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws (LLM) in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"Master of Laws in Alternative Dispute Resolution (LLM in ADR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws in Alternative Dispute Resolution (LLM in ADR).txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Master of Science in Innovation Economics, Law and Regulation (MIELR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Science in Innovation Economics, Law and Regulation (MIELR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Tuition & Financial Aid - Alternative Dispute Resolution Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Alternative Dispute Resolution Certificate.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Tuition & Financial Aid - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - LLM in ADR.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Tuition & Financial Aid - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Tuition & Financial Aid - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Tuition & Financial Aid LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Tuition and Financial Aid Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition and Financial Aid Master of Laws (LLM) - Online.txt',
#             "tags" : "usc",
#             "program": "Online LLM"
#             },
#             {
#             'Title':"Two-Year Extended Master of Laws (LLM)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Two-Year Extended Master of Laws (LLM).txt',
#             "tags" : "usc",
#             "program": "Two-Year LLM"
#             },
#             {
#             'Title':"Types of Aid - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Types of Aid - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"USC Gould -  Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould -  Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"USC Gould - Centre for Dispute Resolution_",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould - Centre for Dispute Resolution_.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             # {
#             # 'Title':"USC Gould - Housing Flyer",
#             # 'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             # 'filename' :'files/USC Gould - Housing Flyer.txt',
#             # "tags" : "usc",
#             # "program": ""
#             # },
#             {
#             'Title':"USC Gould - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould - Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Alternative Dispute Resolution (ADR) Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Alternative Dispute Resolution (ADR) Certificate.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Application Instructions - Alternate Dispute Resolution (ADR) Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Alternate Dispute Resolution (ADR) Certificate.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Application Instructions - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Application Instructions - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Application Instructions - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Application Instructions Extended Master of Laws (LLM)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Extended Master of Laws (LLM).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Application Instructions LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in ADR.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Application Instructions LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Application Instructions LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"Application Instructions Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Application Instructions Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Laws (LLM) - Online.txt',
#             "tags" : "usc",
#             "program": "Online LLM"
#             },
#             {
#             'Title':"Application Instructions Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Laws (LLM) Degree.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Bar and Certificate Tracks Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Bar and Certificate Tracks Master of Laws (LLM) - Online.txt',
#             "tags" : "usc",
#             "program": "Online LLM"
#             },
#             {
#             'Title':"Campus Housing - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Campus Housing - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in ADR.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Campus Housing - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Campus Housing - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"Campus Housing - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Campus Housing - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Campus Housing - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Career and Bar - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc",
#             "program": ""
#             },
#             {
#             'Title':"Career and Bar - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in ADR.txt',
#             "tags" : "usc",
#             "program": "ADR"
#             },
#             {
#             'Title':"Career and Bar - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in International Business and Economic Law.txt',
#             "tags" : "usc",
#             "program": "IBEL"
#             },
#             {
#             'Title':"Career and Bar - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc",
#             "program": "PLCS"
#             },
#             {
#             'Title':"Career and Bar Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar Master of Comparative Law (MCL).txt',
#             "tags" : "usc",
#             "program": "MCL"
#             },
#             {
#             'Title':"Career Opportunities - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career Opportunities - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc",
#             "program": ""
#             },
#         ]


# processor.process_documents(dataset)

# Example query
# result = processor.query_index("Why Choose the 1-Year LLM at USC Gould?", top_k=5, filter_filename="LLM - 1 year.txt")
# print(result)

# import os

# from groq import Groq

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of low latency LLMs",
#         }
#     ],
#     model="mixtral-8x7b-32768",
# )

# print(chat_completion.choices[0].message.content)
