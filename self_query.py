import os
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from dotenv import load_dotenv

# os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI embeddings
load_dotenv()
embeddings = OpenAIEmbeddings()
# Define documents

docs = [
    Document(
        page_content="Complex, layered, rich red with dark fruit flavors",
        metadata={"name":"Opus One", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
        metadata={"name":"Château d'Yquem", "year": 2015, "rating": 98, "grape": "Sémillon", "color":"white", "country":"France"},
    ),
    Document(
        page_content="Full-bodied red with notes of black fruit and spice",
        metadata={"name":"Penfolds Grange", "year": 2017, "rating": 97, "grape": "Shiraz", "color":"red", "country":"Australia"},
    ),
    Document(
        page_content="Elegant, balanced red with herbal and berry nuances",
        metadata={"name":"Sassicaia", "year": 2016, "rating": 95, "grape": "Cabernet Franc", "color":"red", "country":"Italy"},
    ),
    Document(
        page_content="Highly sought-after Pinot Noir with red fruit and earthy notes",
        metadata={"name":"Domaine de la Romanée-Conti", "year": 2018, "rating": 100, "grape": "Pinot Noir", "color":"red", "country":"France"},
    ),
    Document(
        page_content="Crisp white with tropical fruit and citrus flavors",
        metadata={"name":"Cloudy Bay", "year": 2021, "rating": 92, "grape": "Sauvignon Blanc", "color":"white", "country":"New Zealand"},
    ),
    Document(
        page_content="Rich, complex Champagne with notes of brioche and citrus",
        metadata={"name":"Krug Grande Cuvée", "year": 2010, "rating": 93, "grape": "Chardonnay blend", "color":"sparkling", "country":"New Zealand"},
    ),
    Document(
        page_content="Intense, dark fruit flavors with hints of chocolate",
        metadata={"name":"Caymus Special Selection", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Exotic, aromatic white with stone fruit and floral notes",
        metadata={"name":"Jermann Vintage Tunina", "year": 2020, "rating": 91, "grape": "Sauvignon Blanc blend", "color":"white", "country":"Italy"},
    ),
]
# Create vectorstore
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata field information
metadata_field_info = [
    AttributeInfo(
        name="grape",
        description="The grape used to make the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="name",
        description="The name of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="color",
        description="The color of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the wine was released",
        type="integer",
    ),
    AttributeInfo(
        name="country",
        description="The name of the country the wine comes from",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="The Robert Parker rating for the wine 0-100", type="integer" #float
    ),
]
document_content_description = "Brief description of the wine"

# Initialize OpenAI language model
openai_api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(temperature=0.6, openai_api_key=openai_api_key)

# Create retriever with limit enabled
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# Retrieve relevant documents with a specified limit and store the data
data = retriever.get_relevant_documents("what are two that have a rating above 97")
print("data : ", data)
