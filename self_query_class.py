# import os
# from langchain.schema import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import OpenAI
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains.query_constructor.base import AttributeInfo
# from dotenv import load_dotenv
# from files.dataset import docs
# import time
# class DocumentRetriever:
#     def __init__(self):
#         load_dotenv()
#         self.embeddings = OpenAIEmbeddings()
#         self.openai_api_key = os.environ['api_key']
#         self.vectorstore = self._create_vectorstore()
#         self.llm = self._initialize_llm()

#     def _create_vectorstore(self):
#         return Chroma.from_documents(docs, self.embeddings)

#     def _initialize_llm(self):
#         return OpenAI(temperature=0.6, openai_api_key=self.openai_api_key)

#     def _get_metadata_field_info(self):
#         return [
#             AttributeInfo(
#                 name="title",
#                 description="The title of the document",
#                 type="string",
#             ),
#             AttributeInfo(
#                 name="URL",
#                 description="The URL of the document",
#                 type="string",
#             ),
#             AttributeInfo(
#                 name="filename",
#                 description="The filename of the document",
#                 type="string",
#             ),
#             AttributeInfo(
#                 name="tags",
#                 description="Tags associated with the document",
#                 type="string",
#             ),
#             AttributeInfo(
#                 name="content",
#                 description="The content of the document",
#                 type="string",
#             )
#         ]

#     def retrieve_documents(self, query, limit=4):
#         metadata_field_info = self._get_metadata_field_info()
#         document_content_description="Related information about USC Gould Law School"
#         retriever = SelfQueryRetriever.from_llm(
#             self.llm,
#             self.vectorstore,
#             document_content_description,
#             metadata_field_info,
#             enable_limit=True,
#             verbose=True,
#         )
#         return retriever.get_relevant_documents(query, k=limit)
        
