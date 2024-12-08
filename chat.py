# from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import os
import utils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

load_dotenv(find_dotenv())
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), ".credentials", "devc-poc-f047a4fe7fc9.json")

def auth_and_load_folder(folder_id):
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        # credentials_path=os.path.join(os.getcwd(), "credentials.json"),
        # token_path=os.path.join(os.getcwd(), "token.json")
    )
    docs = loader.load()
    
    return docs

def create_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )

    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    return retriever

def create_chain(llm, retriever):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

# if __name__ == "__main__":
#     folder_url = utils.extract_folder_id(input("Folder: "))
#     docs = auth_and_load_folder(folder_url)
#     retriever = create_retriever(docs)
#     llm = ChatOpenAI(
#         temperature=0, 
#         model_name="gpt-3.5-turbo", 
#         api_key=os.getenv("OPENAI_API_KEY")
#     )
#     qa = create_chain(llm, retriever)
#     while True:
#         query = input("> ")
#         answer = qa.run(query)
#         print(answer)
