from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import json
import os
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
        self.embeddings = OpenAIEmbeddings()
        
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
                    'url': file['URL']
                }
                id = file['filename'] + str(i)
                upsert_response = self.index.upsert(vectors=[
                    {
                        "id": id,
                        "values": vec,
                        "metadata": metadata
                    }
                ])
                
    def query_index(self, query, top_k=10, filter_filename=None):
        query_vector = processor.model.encode(query).tolist()
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            # include_values=True,
            include_metadata=True,
            filter={"Title": {"$eq": filter_filename}} if filter_filename else None
        )
        print(result)
        modifiedData = ""
        for doc in  result.matches:
            modifiedData  += json.dumps(doc.metadata)
        return modifiedData

# Instantiate the PineconeDocumentProcessor class
processor = PineconeDocumentProcessor()

# dataset =[
            
#             {
#             'Title':"Career Opportunities - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career Opportunities - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - Alternative Dispute Resolution Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Alternative Dispute Resolution Certificate.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in ADR.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum - Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum - Master of Laws (LLM) - Online.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Curriculum Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Curriculum Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"LLM - 1 year",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/LLM - 1 year.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of Comparative Law (MCL) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Comparative Law (MCL) Degree.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws (LLM) - Online.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of Laws (LLM) in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws (LLM) in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of Laws in Alternative Dispute Resolution (LLM in ADR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Laws in Alternative Dispute Resolution (LLM in ADR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Master of Science in Innovation Economics, Law and Regulation (MIELR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Master of Science in Innovation Economics, Law and Regulation (MIELR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid - Alternative Dispute Resolution Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Alternative Dispute Resolution Certificate.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - LLM in ADR.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition & Financial Aid LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition & Financial Aid LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Tuition and Financial Aid Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Tuition and Financial Aid Master of Laws (LLM) - Online.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Two-Year Extended Master of Laws (LLM)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Two-Year Extended Master of Laws (LLM).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Types of Aid - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Types of Aid - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"USC Gould -  Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould -  Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"USC Gould - Centre for Dispute Resolution_",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould - Centre for Dispute Resolution_.txt',
#             "tags" : "usc"
#             },
#             # {
#             # 'Title':"USC Gould - Housing Flyer",
#             # 'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             # 'filename' :'files/USC Gould - Housing Flyer.txt',
#             # "tags" : "usc"
#             # },
#             {
#             'Title':"USC Gould - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/USC Gould - Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Alternative Dispute Resolution (ADR) Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Alternative Dispute Resolution (ADR) Certificate.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions - Alternate Dispute Resolution (ADR) Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Alternate Dispute Resolution (ADR) Certificate.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions Extended Master of Laws (LLM)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Extended Master of Laws (LLM).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in ADR.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Laws (LLM) - Online.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Application Instructions Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Application Instructions Master of Laws (LLM) Degree.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Bar and Certificate Tracks Master of Laws (LLM) - Online",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Bar and Certificate Tracks Master of Laws (LLM) - Online.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in ADR.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Campus Housing - Master of International Trade Law and Economics (MITLE)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Campus Housing - Master of International Trade Law and Economics (MITLE).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career and Bar - 1 yr Master of Laws (LLM) Degree",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - 1 yr Master of Laws (LLM) Degree.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career and Bar - LLM in ADR",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in ADR.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career and Bar - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in International Business and Economic Law.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career and Bar - LLM in Privacy Law and Cybersecurity",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar - LLM in Privacy Law and Cybersecurity.txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career and Bar Master of Comparative Law (MCL)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career and Bar Master of Comparative Law (MCL).txt',
#             "tags" : "usc"
#             },
#             {
#             'Title':"Career Opportunities - Master of Dispute Resolution (MDR)",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'files/Career Opportunities - Master of Dispute Resolution (MDR).txt',
#             "tags" : "usc"
#             },
#         ]


# processor.process_documents(dataset)

# Example query
# result = processor.query_index("Why Choose the 1-Year LLM at USC Gould?", top_k=2, filter_filename="LLM - 1 year.txt")
# print(result)
