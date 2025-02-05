from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
#from langserve import add_routes
import uvicorn
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2:1b")

# Define the prompt template using from_messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries or explain about the given topic."),
        ("user", "Question or Topic: {question}")
    ]
)

@app.post("/invoke")
async def invoke(request: dict):
    question = request.get('input', {}).get('question', '')
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    # Construct the input for the LLM model
    response = prompt | llm
    output = response.invoke({"question": question})

    # Return the output as a JSON response
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
