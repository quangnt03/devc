import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
# from google.oauth2 import service_account
import os
# import re
from chat import auth_and_load_folder, create_chain, create_retriever
import utils

# Load environment variables
load_dotenv(find_dotenv())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
    # st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
if "google_drive_folders" not in st.session_state:
    st.session_state["google_drive_folders"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

# Set up credentials for Google Drive
# SERVICE_ACCOUNT_FILE = os.path.join(".", ".credentials", "service-account-key.json")  # Replace with your service account key file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if "llm" not in st.session_state:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    st.session_state["llm"] = llm

# Sidebar for Google Drive folder management
with st.sidebar:
    st.header("Add Google Drive Folder")
    folder_url = st.text_input("Enter Google Drive Folder URL:")
    if st.button("Add"):
        if folder_url:
            folder_id = utils.extract_folder_id(folder_url)
            if folder_id:
                docs = auth_and_load_folder(folder_id)
                retriever = create_retriever(docs)
                if retriever:
                    st.session_state["retriever"] = retriever
                    st.session_state["google_drive_folders"].append(folder_id)
                    st.success(f"Folder {folder_id} added successfully!")
            else:
                st.warning("Invalid Google Drive folder URL. Please check the URL.")
        else:
            st.warning("Please enter a valid folder URL.")

    # Display added folders
    if st.session_state["google_drive_folders"]:
        st.subheader("Added Folders:")
        for folder in st.session_state["google_drive_folders"]:
            st.write(folder)


# Main app title

st.title("ChatGPT-like Clone: Echo Bot")

# Set a default model (optional, can be expanded later)
# if len(st.session_state["google_drive_folders"])
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    print(st.session_state)
    if "google_drive_folders" in st.session_state and len(st.session_state["google_drive_folders"]) and st.session_state["retriever"]:
        # Generate echo response
        llm = st.session_state["llm"]
        retriever = st.session_state["retriever"]
        qa = create_chain(llm, retriever=retriever)
        gen = qa.invoke({"query": prompt})
        response = gen["result"]
    else:
        st.session_state["google_drive_folders"] = []
        response = "No Google Drive folder has been added. Please add a folder to enable document retrieval."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)