
import streamlit as st
import os
import faiss
import numpy as np
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModel
from googletrans import Translator
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Arabic BERT model
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModel.from_pretrained("asafaya/bert-base-arabic")

# FAISS Index
index = None
document_chunks = []


if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = {"input": [], "output": []}
    
PROMPT_TEMPLATE = """
استند إلى السياق التالي للإجابة على السؤال بدقة وباللغة العربية فقط. 
إذا لم تجد معلومات كافية في السياق، قم بتقديم تحليل عام وفق معرفتك القانونية.

السياق:
{context}

السؤال: {question}

إجابة مختصرة وواضحة:
"""

def translate(text, dest_lang="ar"):
    """Translate text using Google Translate."""
    try:
        translator = Translator()
        return translator.translate(text, src="auto", dest=dest_lang).text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Fallback to original

def get_pdf_text(files):
    """Extract text from uploaded PDFs."""
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text):
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)

def embed_texts(texts):
    """Generate embeddings using Hugging Face model with [CLS] token."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings.numpy().astype("float32")


def process_documents(uploaded_files):
    """Process PDFs and store embeddings in FAISS."""
    global index, document_chunks
    text = get_pdf_text(uploaded_files)
    chunks = split_text(text)
    document_chunks = chunks  
    
    embeddings = embed_texts(chunks)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])  
    index.add(embeddings) 
    st.sidebar.success(f"✅ Processed {len(uploaded_files)} document .")
    return index 

def retrieve_from_faiss(query,files, k=10):
    """Retrieve top-k relevant document chunks from FAISS."""
    index=process_documents(files)
    if index is None:
        st.warning("⚠️ No documents processed yet!")
        return None
    
    query_embedding = embed_texts([query])
    distances, idxs = index.search(query_embedding, k)


    return [document_chunks[i] for i in idxs[0] if i < len(document_chunks)]

def run_llama3(prompt):
    """Run the Llama3 model."""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )

        return result.stdout.strip() if result.stdout.strip() else "No answer"
    except subprocess.CalledProcessError as e:
        return f"Error occurred: The model is unavailable due to resource constraints"
    except Exception as e:
        return f"Unexpected error: {str(e)}"



def generate_answer(query):
    """Retrieve relevant text and generate an answer with Llama3."""
    translated_query = translate(query, "ar")
    
    results = retrieve_from_faiss(translated_query, uploaded_files)

    context_text = "\n\n---\n\n".join(results) if results else "No context is found"

    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
    history_text = "\n".join([f"Q: {msg.content}" if isinstance(msg, HumanMessage) else f"A: {msg.content}" for msg in chat_history])
    
    prompt = PROMPT_TEMPLATE.format(context=f"{context_text}\n\nConversation History:\n{history_text}", question=translated_query)
    
    # Get Arabic response
    response = run_llama3(prompt)

    translator = Translator()
    response= translator.translate(response, src="auto", dest="en").text
    
    st.session_state.conversation_history["input"].append(query)
    st.session_state.conversation_history["output"].append(response)

    # Save context to memory
    st.session_state.memory.save_context({"input": query}, {"output": response}) 
    return response  if response else "No correspend answer"

# Streamlit UI
st.set_page_config(page_title="Info Extraction", page_icon=":books:", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Arabic PDF Semantic Search + AI Answers</h1>", unsafe_allow_html=True)

query_text = st.text_input("here", placeholder="📖 Ask a Question About Your Documents",label_visibility="collapsed")

with st.sidebar:
    st.subheader("📂 Your Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        with st.progress(" Processing documents..."):
            process_documents(uploaded_files)

if query_text and st.button("Answer"):
    with st.spinner("🔎 Searching and Generating Response..."):
        response = generate_answer(query_text)

    if response:
        st.subheader("📝 AI Answer:")
        st.write(response)
        
    else:
        st.warning("No response generated.")
        

    with st.expander("💬 Conversation History:"):
        chat_data = st.session_state.memory.load_memory_variables({})
        chat_history = chat_data.get("chat_history", [])

        if chat_history:
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    st.write(f"**Q:** {message.content}")  
                elif isinstance(message, AIMessage):
                    st.write(f"**A:** {message.content}") 
        else:
            st.write("No Conversation History is Available")

 

