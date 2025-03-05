import logging
import pathlib
from langchain_core.documents import Document
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import (
  PyPDFLoader, TextLoader,
  UnstructuredWordDocumentLoader,
  UnstructuredEPubLoader
)
from langchain_community.vectorstores import docarray
from langchain.vectorstores.faiss import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler
from ctransformers import AutoModelForCausalLM
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from transformers import LlamaForCausalLM, LlamaTokenizer
import pickle
import os
import time
import tempfile
import pypdfium2 as pdfium
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline



def load_document(temp_filepath: str) -> list[Document]:
    """In case the file is not present, load a file and return it as a list of documents."""
    temp_filepath_chck = os.listdir(temp_filepath)[0]
    pdf = pdfium.PdfDocument(temp_filepath)
    n_pages = len(pdf)
    langchain_docs = []
    content_docs = []
    raw_text = ''
    for i in range(n_pages):
        if i>=2:
            page = pdf[i]
            text = page.get_textpage().get_text_bounded()
            metadata = {"page": i + 1, "source": temp_filepath}

            if text.strip():  # Avoid empty pages
                langchain_docs.append(Document(page_content=text, metadata=metadata))
                content_docs.append(text)
                raw_text+=text
    
    return langchain_docs


def configure_retriever(docs: list[Document], use_compression: bool = True) -> BaseRetriever:
    """Retriever to use."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    if not use_compression:
        return retriever
    
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )


def configure_retriever_local(use_compression: bool = False) -> BaseRetriever:
    """Retriever to use."""
    embeddings_hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    load_path = 'vector_store/retr_mistral_clean1'
    vectordb = FAISS.load_local(folder_path = load_path, embeddings = embeddings_hf, allow_dangerous_deserialization = True)
    
    retriever = vectordb.as_retriever(search_kwargs={"k":10})

    if not use_compression:
        return retriever
    

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings_hf, similarity_threshold=0.76
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )



def configure_chain(retriever: BaseRetriever) -> Chain:
    """Configure chain with a retriever."""

    
    # Model name
    model_name = "raisinghanijayant/doctor-her2-chat-finetune"

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    gpu_chck = 0
    if torch.cuda.is_available():
        gpu_id = 0  # Change if multiple GPUs
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory = total_memory/1000000000.0
        if total_memory > 8:
            gpu_chck = 1
    if gpu_chck ==1:
        device = torch.device("cuda")
        model.to(device)

    # Function to generate responses
    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to GPU
        output = model.generate(**inputs, max_length=512)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Wrap model in Hugging Face pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

    # Store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    

    # Use HuggingFacePipeline as LLM in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)


    
    

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
        
)

def load_css():
    with open("style.css") as f: 
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html= True)


def chatbot_ui(qa_chain):
    load_css()
    st.image("images/logo2.png")
    st.title('LANGCHAIN DEMO')

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {"response_times": [], "satisfaction": []}
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "user_response" not in st.session_state:
        st.session_state.user_response = ""

    # Preserve chat input across interactions
    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.user_query = user_query  # ✅ Store query in session state
        
        start_time = time.time()
        stream_handler = StreamlitCallbackHandler(st.chat_message("assistant"))
        response_dict = qa_chain.invoke({"question": user_query})
        response_time = time.time() - start_time
        #print(response["answer"])
        response = response["answer"].split("Helpful Answer:")[-1].strip()
        st.session_state.user_response = response

        # Store chat history
        st.session_state.chat_history.append((user_query, response))
        st.session_state.metrics["response_times"].append(response_time)

        # Store feedback as None initially
        feedback_key = len(st.session_state.chat_history) - 1
        st.session_state.feedback[feedback_key] = None  # Placeholder for feedback
        #st.markdown(response)
    #if st.session_state.user_query:
        #st.write(f"**You:** {st.session_state.user_query}")

    with st.container():
        st.write(st.session_state.user_response)

        feedback_key = len(st.session_state.chat_history) - 1
        if feedback_key not in st.session_state.feedback:
            st.session_state.feedback[feedback_key] = None  # Ensure key exists

        # ✅ Buttons inside user_query but no reset
        if st.session_state.user_query != "" :
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 Yes", key=f"yes_{feedback_key}"):
                    st.session_state.feedback[feedback_key] = 1
                    st.session_state.metrics["satisfaction"].append(1000)
                    st.session_state.user_query = ""
                    st.rerun()
            with col2:
                if st.button(f"👎 No", key=f"no_{feedback_key}"):
                    st.session_state.feedback[feedback_key] = 0
                    st.session_state.metrics["satisfaction"].append(0)
                    st.session_state.user_query =""
                    st.rerun()

        

    # Display chat history
    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for idx, message in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f'<div class="user-message">{message[0]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{message[1]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Metrics tab
def metrics_ui():
    load_css()
    ##streamlit framework - integrate our framework with OpenAI 
    st.image("images/logo1.png", width= 500)
    st.title("Chatbot Metrics")
    if "metrics" in st.session_state:
        avg_response_time = sum(st.session_state.metrics["response_times"]) / max(len(st.session_state.metrics["response_times"]), 1)
        avg_satisfaction = sum(st.session_state.metrics["satisfaction"]) 
        #/ max(len(st.session_state.metrics["satisfaction"]), 1) * 100 
        
        st.write(f"📊 **Average Response Time:** {avg_response_time:.2f} seconds")
        st.write(f"😊 **User Satisfaction:** {avg_satisfaction:.2f}%")
    else:
        st.write("No metrics available yet.")


# Main function
def main():

    retr = configure_retriever_local()
    qa_chain = configure_chain(retriever=retr)
    
    
    tab1, tab2 = st.tabs(["Chatbot", "Metrics"])
    with tab1:
        chatbot_ui(qa_chain)
    with tab2:
        metrics_ui()

if __name__ == "__main__":
    main()