import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import os
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

st.title("Customer Support")
uploaded_file = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
try:
    loader = CSVLoader(file_path=uploaded_file, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db = FAISS.from_documents(data, embeddings)
if not os.path.exists(DB_FAISS_PATH):
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index created and saved at {DB_FAISS_PATH}")
else:
    print(f"FAISS index already exists at {DB_FAISS_PATH}, skipping creation.")

llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey !"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
