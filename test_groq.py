from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("API KEY NOT FOUND! Check .env")
else:
    print("API KEY LOADED SUCCESSFULLY!")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    response = llm.invoke(
        "Explain RAG in two simple sentences."
    )

    print("\nGroq Response:")
    print(response.content)