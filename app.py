from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import streamlit as st
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# LangSmith / LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize local model (Ollama)
llm = ChatOllama(model="llama3")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please response to the user queries."),
    ("user", "{input}")
])

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Streamlit UI
st.title("Local Chatbot with LangChain Tracing")

user_input = st.text_input("Ask something:")

if user_input:
    response = chain.invoke({"input": user_input})
    st.write(response)