import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
load_dotenv()

 # Configuration

PDF_PATH  = "data/sample.pdf"
VECTOR_DB_PATH = "./vectorstore"
COLLECTION_NAME = "documind"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Load PDF

print("\n[1/6] Loading PDF..")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
if not documents:
    raise ValueError(f"No documents found in the PDF: {PDF_PATH}")
print(f"loaded Sucessfully {len(documents)}pages")

# Chunking

print("\n[2/6] Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
if not chunks:
    raise ValueError("No chunks were created from the documents.")
print(f"Chunked into {len(chunks)} chunks") 

# Embedding

print("\n[3/6] Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
   model= "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding created successfully.")

# Vector Database ( chroma DB)

print("\n[4/6] Creating vector database...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_PATH
)

print("Documents embedded and stored in vector database successfully.")

#initisalise Groq 

print("\n[5/6] Initializing Groq...")
llm = ChatGroq(
   model  = "openai/gpt-oss-20b",
   temperature = 0
)
print("Groq initialized successfully.")

# RETREIVAL AND GENERATION 

def ask_question(question):
   print("\n[6/6] Retrieving relevant documents and generating answer...")
   retrieved_docs = vector_db.similarity_search(question, k=3)
   if not retrieved_docs:    
       raise ValueError("No relevant sections found for the question.")
       return


contex = "\n\n".join([doc.page_content for doc in retrieved_docs])
prompt = f"""
You are a helpful Assistant . Answer only the question below and . If the answer is not present in the context below , say : " The information is not available in the provided documents."

CONTEXT:{context}
QUESTION:{question}
ANSWER:
"""

response = llm.invoke(prompt)
print(f"\nAnswer: {response}")
print(response.content)

print("sources")
for doc in retrieved_docs:
page = doc.metadata.get("page", "Unknown page")
print(f"page : {page + 1 if isinstance(page, int) else page}")

# LOOP FOR USER INPUT

print("Your PDF is ready to be queried. You can ask questions about the content of the PDF.")
print("Type 'exit' to stop.\n")

while True:
    user_question = input("Enter your question: ")
    if user_question.lower() == "exit":
         print("Exiting the program.")
         break
    try:
         ask_question(user_question)
    except Exception as e:
         print(f"An error occurred: {e}")

