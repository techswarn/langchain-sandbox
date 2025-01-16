# importing the modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import bs4
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# For loading documents from web
from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio
nest_asyncio.apply()
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_transformers import BeautifulSoupTransformer
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
import re

#transformer pipeline 
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
store = {}
#Create LLM instance
def llm_instance():
    model_id = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_vectorstore_from_url():

    
   # Load the web page content
    loader = WebBaseLoader(["https://docs.digitalocean.com/products/app-platform/"])
    loader.requests_per_second = 1
    docs = loader.load()  

    # Use Beautifu soup to extract the page content
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs)
    docs_content = docs_transformed[0].page_content
    
    # Remove HTML like tags and unnecessary backslashes
    clean_text = re.sub(r'<\\?/?[a-zA-Z]+>', '', docs_content)  # Remove HTML-like tags
    clean_text = re.sub(r'\\/[wp:paragrph]', ' ', clean_text)  # replace /wp:paragrph
    clean_text = re.sub(r'\\[nr]', ' ', clean_text)  # Replace \n, \r with space
    clean_text = re.sub(r'\\/', '/', clean_text)  # Fix escaped forward slashes

    # Decode unicode escape sequences
    clean_text = bytes(clean_text, "utf-8").decode("unicode_escape")

    # Remove additional extraneous characters and whitespace
    clean_text = re.sub(r'["]{2,}', '"', clean_text)  # Reduce multiple quotes
    clean_text = re.sub(r'\s{2,}', ' ', clean_text)  # Reduce excessive spaces

   # Reduce the number of tokens to exract only the desired content
    data = clean_text[:6891]

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[''],
        chunk_size=256,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([data])
    
    # Create a vector store from the chunks
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   vector_store = Chroma.from_documents(document_chunks, embedding=embeddings)

    # Load the data into the vector store and set the retriever
    #vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    return vector_store

# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. Your name is bubble"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Make sure you answer in minimum 20 sentence"
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



def main():
    print("-----Loading the LLM----- ")
    llm = llm_instance()
    print("-----Loading data from Docs and creating vector store------")
    vectorStore = get_vectorstore_from_url()
    retriever = vectorStore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )   

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )   

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

 

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ### Statefully manage chat history ###
    store = {}


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )   


    # # defining the loop for a conversation with the AI
    while True:
        question = input("Enter your query: ")
        if question == 'exit': 
            break 
        # getting the response

        response = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"] 

        print(response)
    


if __name__ == "__main__":
    main()