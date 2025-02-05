# -*- coding: utf-8 -*-
"""RAG LLM Chatbot.ipynb

"""

import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=""
os.environ["LANGCHAIN_PROJECT"]="rag-llm-chatbot"

!pip install -q --upgrade langchain langchain-openai langchain-core langchain_community langchain_huggingface transformers accelerate bitsandbytes docx2txt pypdf langchain_chroma sentence_transformers

!huggingface-cli login

import langchain
print(langchain.__version__)

import os
os.environ["HF_TOKEN"] = "hf_token"

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.chains import ConversationChain
import transformers
import torch
import warnings
warnings.filterwarnings('ignore')

model="meta-llama/Llama-3.2-1B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model)
pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True
    )

llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.7})

llm_response = llm.invoke("Tell me a joke")
print(llm_response)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(llm_response)

chain = llm | output_parser
chain.invoke("Tell me a joke")

!pip install -U -q bitsandbytes

from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

!pip install --upgrade -q transformers

#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import warnings

warnings.filterwarnings('ignore')

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#                  model_id,
#                  quantization_config=quantization_config
#                  )

# Create a text generation pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    #torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True,
    repetition_penalty=1.03,
    return_full_text=False
)

# ✅ Wrap the pipeline inside HuggingFacePipeline
hf_llm = HuggingFacePipeline(pipeline=chat_pipeline, model_kwargs={'temperature': 0.7, "quantization_config": quantization_config})

# ✅ Pass it to ChatHuggingFace
chat_model = ChatHuggingFace(llm=hf_llm)

# Test conversation
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = chat_model.invoke(messages)
print(response.content)  # Expected Output: "The capital of France is Paris."

response = chat_model.invoke("Tell me a joke")
print(response.content)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(response)

chain = chat_model | output_parser
chain.invoke("Tell me a joke")

!ps aux --sort=-%mem | head -10

import os
os._exit(0)  # ✅ Forces a kernel restart

from typing import List
from pydantic import BaseModel, Field

class MobileReview(BaseModel):
  phone_model: str = Field(description="Name and model of the phone")
  rating: float = Field(description="Overall rating out of 5:")
  pros: List[str] = Field(description="List of positive aspects")
  cons: List[str] = Field(description="List of negative aspects")
  summary: str = Field(description="Brief summary of the review")

review_text = """
Just got my hands on the new Galaxy S23 and wow, this thing is slick! The screen is gorgeous,
colors pop like crazy, Camera's insane too, especially at night - my Insta game's never been
stronger. Battery life's solid, lasts me all day no probelm.

Not gonna lie though, it's pretty pricey. And what's with ditching the charger? C'mon samsung.
Also, still getting used to the new button layout, keep hitting Bixby by mistake.

Overall, I'd say it's a solid 4 out of 5. Great phone, but a few annoying quirks keep it from
being perfect. If you're due for an upgrade, definitely worth checking out!
"""

structured_llm = chat_model.with_structured_output(MobileReview)
structured_llm.invoke(review_text)

structured_prompt = f"""
Extract structured information from the following review:

Review:
{review_text}

Format your response as JSON with the following fields:
- phone_model: (Phone name and model)
- rating: (Overall rating out of 5)
- pros: (List of positive aspects)
- cons: (List of negative aspects)
- summary: (Brief summary of the review)

JSON Output:
"""
output = chat_model.invoke(structured_prompt)
print(output)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
prompt.invoke({"topic": "programming"})

chain = prompt | chat_model | output_parser
chain.invoke({"topic": "car drivers"})

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

system_message = SystemMessage(content="You are a helpful assistant that tells jokes.")
human_message = HumanMessage(content="Tell me about programming")
chat_model.invoke([system_message, human_message])

template = ChatPromptTemplate([
    ("system","You are a helpful assistant that tells jokes."),
    ("human","Tell me about {topic}")
])

prompt_value = template.invoke({"topic": "programming"})
print(prompt_value)

chat_model.invoke(prompt_value)

response = chat_model.invoke(prompt_value)
print(response.content)

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

docx_loader = Docx2txtLoader("/content/docs/GreenGrow Innovations_ Company History.docx")
documents = docx_loader.load()

print(len(documents))

splits = text_splitter.split_documents(documents)

print(f"Split the documents into {len(splits)} chunks.")

documents[0]

splits[0]

splits[0].metadata

splits[0].page_content

# Function to load documents from a folder

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

# Load documents from a folder
folder_path = "/content/docs"
documents = load_documents(folder_path)

print(f"Loaded {len(documents)} documents from the folder.")
splits = text_splitter.split_documents(documents)
print(f"Split the documents into {len(splits)} chunks.")

embeddings = HuggingFaceEmbeddings()

# Embedding Documents

document_embeddings = embeddings.embed_documents([split.page_content for split in splits])

print(f"Created embeddings for {len(document_embeddings)} document chunks.")

document_embeddings[0]

# !pip install sentence_transformers
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
document_embeddings[0]

# Chroma DB
from langchain_chroma import Chroma

embedding_function = HuggingFaceEmbeddings()
collection_name = "my_collection"
vectorstore = Chroma.from_documents(collection_name=collection_name, documents=splits, embedding=embedding_function, persist_directory="./chroma_db")
#db.persist()
print("Vector store created and persisted to './chroma_db'")

# Perform similarity search

query = "When was GreenGrow Innovations founded?"
search_results = vectorstore.similarity_search(query, k=2)

print(f"\nTop 2 most relevant chunks for the query: '{query}'\n")
for i, result in enumerate(search_results, 1):
    print(f"Result {i}:")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Content: {result.page_content}")
    print()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever.invoke("When was GreenGrow Innovations founded?")

from langchain_core.prompts import ChatPromptTemplate
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt
)
rag_chain.invoke("When was GreenGrow Innovations founded?")

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()} | prompt
)
rag_chain.invoke("When was GreenGrow Innovations founded?")

from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)
question = "When was GreenGrow Innovations founded?"
response = rag_chain.invoke(question)
print(response)

# Conversational RAG
from langchain_core.messages import HumanMessage, AIMessage
chat_history = []
chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=response)
])

chat_history

from langchain_core.prompts import MessagesPlaceholder
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# history_aware_retriever = create_history_aware_retriever(
#     chat_model, retriever, contextualize_q_prompt
# )
contextualize_chain = contextualize_q_prompt | chat_model | StrOutputParser()
contextualize_chain.invoke({"input": "Where it is headquartered?", "chat_history": chat_history})

from langchain.chains import create_history_aware_retriever
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)
history_aware_retriever.invoke({"input": "Where it is headquartered?", "chat_history": chat_history})

retriever.invoke("Where it is headquartered?")

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    #  ("system", "Tell me joke on Programming"),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain.invoke({"input": "Where it is headquartered?", "chat_history":chat_history})

import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

# Initialize the database
create_application_logs()

import uuid
session_id = str(uuid.uuid4())
chat_history = get_chat_history(session_id)
print(chat_history)
question1 = "When was GreenGrow Innovations founded?"
answer1 = rag_chain.invoke({"input": question1, "chat_history":chat_history})['answer']
insert_application_logs(session_id, question1, answer1, "llama3.2:1b")
print(f"Human: {question1}")
print(f"AI: {answer1}\n")

question2 = "Where it is headquartered?"
chat_history = get_chat_history(session_id)
print(chat_history)
answer2 = rag_chain.invoke({"input": question2, "chat_history":chat_history})['answer']
insert_application_logs(session_id, question2, answer2, "llama3.2:1b")
print(f"Human: {question2}")
print(f"AI: {answer2}\n")

session_id = str(uuid.uuid4())
question = "What is GreenGrow"
chat_history = get_chat_history(session_id)
print(chat_history)
answer = rag_chain.invoke({"input": question, "chat_history":chat_history})['answer']
insert_application_logs(session_id, question, answer, "llama3.2:1b")
print(f"Human: {question}")
print(f"AI: {answer}\n")