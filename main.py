from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from pprint import pprint

app = FastAPI()


########Connect to the CHROMADB collection created in build_knowledge_base.py
#Client is connection to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

#turn text to vecto
ef = OllamaEmbeddingFunction(
    model_name = "nomic-embed-text",
    url = "http://localhost:11434",
)

#Table Name in database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


############ User Class 
class ProfileRequest(BaseModel):
    user_id: str
    user_name: str
    text: str

#Cut big paragraph ex: "......" => ["...", "..."]
def split_into_chunks(text):
    chunks = []
    text_plit = text.split("\n\n")
    for line_of_text in text_plit:
        text_clean = line_of_text.strip()
        if text_clean != "":
            chunks.append(text_clean)
    return chunks

##Submiting Profile
@app.post("/documents")
def add_profile(profile: ProfileRequest):
    chunks = split_into_chunks(profile.text)
    ##save ids, metadata for each chunk and save to ChromaDB
    ids = []
    metadatas = []
    for index in range(len(chunks)):
        ids.append(f"{profile.user_id}_chunk{index}")                        # chunk0, chunk1, chunk2...
        metadatas.append({"user_id": profile.user_id, "user_name": profile.user_name, "chunk_index": index})

    collection.delete(where={"user_id": profile.user_id})

    # upsert = add new or insert even when ID already existed
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )
    return {"user_id": profile.user_id, "chunks_added": len(chunks)}


##### API Endpoints
#ASK QUESTION AND Get Response
@app.get("/ask")
def ask(user_id: str, question: str):
    results = collection.query(
        query_texts=[question],
        n_results=2,
        where={
            "user_id": user_id
        }
    )
    chunks = results['documents'][0]
    if not chunks:
        raise HTTPException(status_code=404, detail="Profile text is empty")
    context = "\n\n".join(chunks)
    #create prompt
    augmented_prompt = f"""Use the following context to answer the question.
    If the context doesn't contain relevant information, say so.

    Context:
    {context}

    Question: {question}"""
    response = ollama.chat(
        model = 'qwen2.5:0.5b',
        messages=[{
            'role': "user",
            'content': augmented_prompt
        }],
    )

    return {
        "user_id":user_id,
        "question":question,
        "answer": response["message"]["content"],
        "context_used": chunks,
    }



