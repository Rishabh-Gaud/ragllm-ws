# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from dotenv import load_dotenv
# import os
# from langchain.vectorstores import Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# # import pinecone
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain

# # Load environment variables from .env file
# # load_dotenv()

# # sentence transformer

# from sentence_transformers import SentenceTransformer
# model_name ="sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)
# dataset = [
#               # {
# #             'Title':"Career Opportunities - Master of International Trade Law and Economics (MITLE)",
# #             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
# #             'filename' :'Application Instructions - Alternate Dispute Resolution (ADR) Certificate.txt',
# #             "tags" : "ucla"
# #             },
#             {
#             'Title':"Alternative Dispute Resolution (ADR) Certificate",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'Alternative Dispute Resolution (ADR) Certificate.txt',
#             "tags" : "ucla"
#             },
# #             {
# #             'Title':"Curriculum - LLM in ADR",
# #             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
# #             'filename' :'data/Curriculum - Master of International Trade Law and Economics (MITLE).txt',
# #             "tags" : "ucla"
# #             },
#             {
#             'Title':"Curriculum - LLM in International Business and Economic Law",
#             'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
#             'filename' :'LLM - 1 year.txt',
#             "tags" : "ucla"
#             },
#               ]
# from pinecone import Pinecone
# import time
# pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# # if use_serverless:
# #     spec = ServerlessSpec(cloud='aws', region='us-west-2')
# # else:
# #     # if not using a starter index, you should specify a pod_type too
# #     spec = PodSpec()
# from langchain_pinecone import PineconeVectorStore

# index_name = "rag-pipeline"

# index = pc.Index("rag-pipeline")

# import os

# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter

# # loader = TextLoader("LLM - 1 year.txt")
# # documents = loader.load()
# # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# # docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# for file in dataset:
#     filename = file["filename"]
#     loader = TextLoader(filename)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)
#     vectorData= []
#     for i in range(len(docs)):
#         print(len(docs))
#         vec = model.encode(docs[i].page_content)
#         vectorData.append({
#             "Title": file["Title"],
#             "URL": file["URL"],
#             "Data": vec
#         })
#         metadata = {
#             'title': file['Title'],
#             'data': docs[i].page_content,
#             'filename': file['filename'],
#             'url': file['URL']
#         }
#         id = file['filename']+str(i)
#         print(id)
#         upsert_response = index.upsert(vectors=[
#         {
#             "id": id, 
#             "values": vec, 
#             "metadata": metadata
#         }
#     ],)


# # docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
# # query = "tell me adr certificate"
# # docs = docsearch.similarity_search(query, k = 3)
# # print(docs)
# q = "Why Choose the 1-Year LLM at USC Gould?"
# enc = model.encode(q).tolist()
# top_k = 3
# result = index.query(
#     vector=enc,
#     top_k=2,
#     include_values=True,
#     include_metadata=True,
#     filter={"filename": {"$eq": "example.txt"}}
# )
# print(result)