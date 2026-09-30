import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

#Open file and read file. Save them to variable
with open("profile.txt", "r", encoding="utf-8") as  file:
    text = file.read()

#Cut text into sentence (chunks)
paragraphs = text.split("\n\n")

chunks = []
for paragraph in paragraphs:
    clean_paragraph = paragraph.strip()
    if clean_paragraph != "":
        chunks.append(clean_paragraph)
print("Loaded", len(chunks), "chunks from profile.txt")

##Open (or create) database chromaDB on disk
#Data will be saved into the disk, cannot be lost even if turning of the computer
chroma_client = chromadb.PersistentClient(path="./chroma_db")


##chromadb Turn text into vecto
##ChromaDB will call model "nomic-embed-text" running in Ollama
embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

##Create collection 
collection = chroma_client.get_or_create_collection(
    name="personal_profile",
    embedding_function=embedding_function,
    )


##save ids, metadata for each chunk and save to ChromaDB
ids = []
metadatas = []
for index in range(len(chunks)):
    ids.append("chunk" + str(index))                        # chunk0, chunk1, chunk2...
    metadatas.append({"source": "profile", "chunk_index": index})

# upsert = add new or insert even when ID already existed
collection.upsert(
    ids=ids,
    documents=chunks,
    metadatas=metadatas,
)

print("Added", len(chunks), "chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")