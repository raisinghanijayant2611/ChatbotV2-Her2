import logging
import pathlib
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import docarray
from langchain.vectorstores.faiss import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler
from ctransformers import AutoModelForCausalLM
#from langchain.llms import CTransformers
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
import csv

#####Placeh holder function to provide functionality for the user to upload more documents (not implemented yet)
def load_document(temp_filepath: str) -> list[Document]:
    """In case the file is not present, load a file and return it as a list of documents."""
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
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

    
    # Model name , fine tune from model: https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025/tree/main
    model_name = "raisinghanijayant/doctor-her2-chat-finetune"

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    gpu_chck = 0
    device_map = "cpu"
    if torch.cuda.is_available():
        gpu_id = 0  # Change if multiple GPUs
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory = total_memory/1000000000.0
        if total_memory > 15:
            gpu_chck = 1
    
    if gpu_chck ==1:
        device = torch.device("cuda")
        device_map = "auto"
        
    else:
        device = torch.device("cpu")
    

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
    model.to(device)

    
    # Wrap model in Hugging Face pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.7, max_new_tokens = 100)

    # Store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k = 1, ai_prefix = "" )
    

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
    st.subheader('HUMAN BREST CANCER PUBLICATION CHATBOT')
    st.write("[Click here to visit Publication](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf)")

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
    file_path_pos = "data/feedback_pos.csv"  
    file_path_neg = "data/feedback_neg.csv" 

    if user_query:
        st.session_state.user_query = user_query  # ✅ Store query in session state
        
        start_time = time.time()
        stream_handler = StreamlitCallbackHandler(st.chat_message("assistant"))
        response_dict = qa_chain.invoke({"question": user_query}, {"callbacks": [stream_handler]})
        response_time = time.time() - start_time
        #print(response["answer"])
        response = response_dict["answer"].split("Helpful Answer:")[-1].strip()
        st.markdown(response)
        if len(response) == 0:
            st.session_state.user_response = "No Response"
        else:
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
            st.write("was this answer helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 Yes", key=f"yes_{feedback_key}"):
                    st.session_state.feedback[feedback_key] = 1
                    st.session_state.metrics["satisfaction"].append(1)
                    # List of texts to add as a new row
                    new_row = [str(st.session_state.user_query), str(st.session_state.user_response), "1"]
                    # Open the CSV file in append mode and add the new row
                    with open(file_path_pos, "a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(new_row)
                    st.session_state.user_query = ""
                    st.rerun()
            with col2:
                if st.button(f"👎 No", key=f"no_{feedback_key}"):
                    st.session_state.feedback[feedback_key] = 0
                    st.session_state.metrics["satisfaction"].append(0)
                    new_row = [str(st.session_state.user_query), str(st.session_state.user_response), "0"]
                    # Open the CSV file in append mode and add the new row
                    with open(file_path_neg, "a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(new_row)
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
        avg_satisfaction = sum(st.session_state.metrics["satisfaction"])/max(len(st.session_state.metrics["satisfaction"]),1)
        
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