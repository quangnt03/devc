import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.retrievers import MergerRetriever
from dotenv import load_dotenv, find_dotenv
import os
import chromadb
from chat import auth_and_load_folder, create_chain, create_retriever
import utils

# Load environment variables
load_dotenv(find_dotenv())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
if "google_drive_folders" not in st.session_state:
    st.session_state["google_drive_folders"] = {}
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = []

# Set up OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if "llm" not in st.session_state:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    st.session_state["llm"] = llm

# Sidebar for Google Drive folder management
with st.sidebar:
    st.header("Add Google Drive Folder")
    folder_url = st.text_input("Enter Google Drive Folder URL:")
    if st.button("Add"):
        if folder_url:
            folder_id = utils.extract_folder_id(folder_url)
            if folder_id:
                if folder_id not in st.session_state["google_drive_folders"]:
                    # try:
                    docs = auth_and_load_folder(folder_id)
                    retriever = create_retriever(docs)
                    if retriever:
                        st.session_state["google_drive_folders"][folder_id] = f"Folder {folder_id}"
                        st.session_state["retrievers"].append(retriever)
                        st.success(f"Folder {folder_id} added successfully!")
                    # except Exception as e:
                    #     st.error(f"Failed to add folder: {e}")
                else:
                    st.warning("This folder has already been added.")
            else:
                st.warning("Invalid Google Drive folder URL. Please check the URL.")
        else:
            st.warning("Please enter a valid folder URL.")

    # Display added folders
    if st.session_state["google_drive_folders"]:
        st.subheader("Added Folders:")
        for folder_id, folder_name in st.session_state["google_drive_folders"].items():
            st.write(folder_name)

# Main app title
st.title("ChatGPT with Multiple Google Drive Folders")

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question:"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for retrievers
    if st.session_state["retrievers"]:
        try:
            # Combine retrievers and query
            llm = st.session_state["llm"]
            retrievers = st.session_state["retrievers"]
            merged_retriever = MergerRetriever(retrievers=retrievers)
            response = ""
            qa = create_chain(llm, merged_retriever)
            result = qa.run(query=prompt)
            response += f"{result}\n"
        except Exception as e:
            response = f"Error: {e}"
    else:
        response = "No Google Drive folders added. Please add folders to enable document retrieval."

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
