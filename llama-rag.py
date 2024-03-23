# import os
# from llama_index.core import VectorStoreIndex
# from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.core.node_parser import SentenceSplitter
# import time

# from IPython.display import HTML

# os.environ["OPENAI_API_KEY"] = "sk"
# # print(os.environ["OPENAI_API_KEY"])

# reader = SimpleDirectoryReader(input_files=["example.pdf"])
# documents_jerry = reader.load_data()

# # reader = SimpleDirectoryReader(input_files=["dense_x_retrieval.pdf"])
# # documents_ravi = reader.load_data()

# index = VectorStoreIndex.from_documents(documents=[])

# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=512, chunk_overlap=20),
#     ]
# )

# for document in documents_jerry:
#     document.metadata["user"] = "Jerry"

# nodes = pipeline.run(documents=documents_jerry)
# index.insert_nodes(nodes)
# print(index)


# # for document in documents_ravi:
# #     document.metadata["user"] = "Ravi"

# # nodes = pipeline.run(documents=documents_ravi)
# # index.insert_nodes(nodes)

# jerry_query_engine = index.as_query_engine(
#     filters=MetadataFilters(
#         filters=[
#             ExactMatchFilter(
#                 key="user",
#                 value="Jerry",
#             )
#         ]
#     ),
#     similarity_top_k=3,
# )
# print(jerry_query_engine, "helo")
# # ravi_query_engine = index.as_query_engine(
# #     filters=MetadataFilters(
# #         filters=[
# #             ExactMatchFilter(
# #                 key="user",
# #                 value="Ravi",
# #             )
# #         ]
# #     ),
# #     similarity_top_k=3,
# # )
# start_time = time.time()
# response = jerry_query_engine.query("tell me about one year llm program")
# end_time = time.time()

# print(response.response)
# print(end_time-start_time)
