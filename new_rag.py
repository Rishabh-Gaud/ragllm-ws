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

# load_dotenv()
# # Uncomment the libraries and run them
# # !pip install pypdf langchain-community
# # !pip install sentence-transformers
# # !pip install openai
# # !pip install faiss-cpu
# # !pip install -qU langchain-text-splitters
# # !pip install langchain
# # !pip install PyMuPDF

# # Load the PDF
# with fitz.open("example.pdf") as doc:
#     text = ""
#     for page in doc:
#         text += page.get_text()
# print(text)
# text = text.replace(" ", '')

# # Initialize the text splitter
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

# # Check if chunk embeddings are saved, otherwise compute and save them
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)
# if os.path.exists("chunks_embeddings.pkl"):
#     with open("chunks_embeddings.pkl", "rb") as f:
#         chunks, chunk_embeddings = pickle.load(f)
# else:
#     chunks = text_splitter.split_text(text)
#     chunk_embeddings = [model.encode(chunk) for chunk in chunks]
#     with open("chunks_embeddings.pkl", "wb") as f:
#         pickle.dump((chunks, chunk_embeddings), f)

# # Load chunk embeddings
# if os.path.exists("chunks_embeddings.pkl"):
#     with open("chunks_embeddings.pkl", "rb") as f:
#         chunks, chunk_embeddings = pickle.load(f)

# # Initialize Faiss index
# dimension = chunk_embeddings[0].shape[0]
# faiss_index = faiss.IndexFlatL2(dimension)
# embeddings_array = np.array(chunk_embeddings).astype('float32')
# faiss_index.add(embeddings_array)

# # Initialize OpenAI language model
# llm_openai = OpenAI(temperature=0.3, openai_api_key=os.environ['api_key'])

# # Define the answer function
# def answer(text, model, listing_history):
#     if len(listing_history) >= 5:
#         listing_history = listing_history[len(listing_history) - 5:]

#     prompt_template = PromptTemplate(
#         input_variables=['query_text', 'retrieved', 'listing_history'],
#         template="""Based on the user's query "{query_text}" and the provided relevant information: {retrieved}, your task is to synthesize a clear, accurate, and concise response. Follow these steps:

# 1. Analyze the provided information to understand its relevance to the user's query.
# 2. Construct a response that directly addresses the user's query using the information given. Aim for clarity and brevity.
# 3. If the information does not fully address the user's query, clearly state that the query cannot be completely answered with the available information and suggest the need for additional sources or clarification.
# 4.you can also use the data in {listing_history} for  answering
# Remember, your goal is to provide a helpful and precise answer that is directly based on the provided information. Avoid making assumptions beyond the given content."""
#     )
#     start_time = time.time()
#     query_embedding = model.encode(text)
#     query_embedding = np.array([query_embedding]).astype('float32')
#     k = 10
#     D, I = faiss_index.search(query_embedding, k)
#     retrieved_list = [chunks[i] for i in I[0]]
#     end_time = time.time()
#     chain = LLMChain(llm=llm_openai, prompt=prompt_template)
#     response_stream = chain.run(query_text=text, retrieved=retrieved_list, listing_history=listing_history, stream=True)
#     return response_stream, end_time - start_time

# listing_history = []

# # Main loop to interact with the user
# while True:
#     user_query = input("Enter the query here: ")
#     if user_query == "exit":
#         break
#     else:
#         output, time_consumed = answer(user_query, model, listing_history)
#         print(output)
#         print(time_consumed)
#         listing_history.append(output)
