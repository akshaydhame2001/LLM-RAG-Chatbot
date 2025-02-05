from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from chroma_utils import vectorstore
from typing import List
import os

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
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


# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])



def get_rag_chain():
    history_aware_retriever = create_history_aware_retriever(chat_model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain