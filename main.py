import os
import json
import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.docstore.document import Document

default_collection_name = "example"
embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

qdrant_args = {"path": "qdrant.local"}

qdrant_client = qdrant_client.QdrantClient(**qdrant_args)


def ingest_data():
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    with open('data.jsonp', 'r', encoding='utf-8') as file:
        content_str = file.read()
    lines = content_str.splitlines()
    for line in lines:
        line = json.loads(line)
        for chunk in splitter.split_text(line["content"]):
            source_chunks.append(Document(page_content=chunk, metadata={"source": line["source"]}))

    Qdrant.from_documents(
        documents=source_chunks,
        embedding=embedding,
        collection_name=default_collection_name,
        **qdrant_args
    )


if __name__ == '__main__':
    ingest_data()

    # This should print true and returns false
    collections = qdrant_client.get_collections().collections
    print(any(collection.name == default_collection_name for collection in collections))

    # This fails
    ingest_data()
