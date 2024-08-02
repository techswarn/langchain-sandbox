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


#transformer pipeline 
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
store = {}
#Create LLM instance
def llm_instance():
    model_id = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_vectorstore_from_url():

    
   # Load the web page content
    loader = WebBaseLoader(["https://docs.digitalocean.com/products/app-platform/", "https://docs.digitalocean.com/products/app-platform/getting-started/quickstart/"
                        "https://docs.digitalocean.com/products/app-platform/how-to/create-apps/", "https://docs.digitalocean.com/products/app-platform/how-to/deploy-from-container-images/", "https://docs.digitalocean.com/products/app-platform/how-to/deploy-from-monorepo/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/scale-app/", "https://docs.digitalocean.com/products/app-platform/how-to/add-deploy-do-button/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-components/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-services/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-jobs/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-workers/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/manage-static-sites/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-functions/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-data-storage/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/manage-databases/", "https://docs.digitalocean.com/products/app-platform/details/limits/", "https://docs.digitalocean.com/products/app-platform/how-to/build-run-commands/", "https://docs.digitalocean.com/products/app-platform/how-to/use-environment-variables/"
                        "https://docs.digitalocean.com/products/app-platform/how-to/cache-content/", "https://docs.digitalocean.com/products/app-platform/how-to/change-region/", "https://docs.digitalocean.com/products/app-platform/how-to/change-stack/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/upgrade-buildpacks/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-time-zone/", "https://docs.digitalocean.com/products/app-platform/how-to/add-ip-address/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/manage-domains/", "https://docs.digitalocean.com/products/app-platform/how-to/configure-cors-policies/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-internal-routing/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/url-rewrites/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-health-checks/", "https://docs.digitalocean.com/products/app-platform/how-to/view-logs/", "https://docs.digitalocean.com/products/app-platform/how-to/forward-logs/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/create-alerts/", "https://docs.digitalocean.com/products/app-platform/how-to/view-insights/",
                        "https://docs.digitalocean.com/products/app-platform/how-to/manage-deployments/", "https://docs.digitalocean.com/products/app-platform/how-to/update-app-spec/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-source-repo/", "https://docs.digitalocean.com/products/app-platform/how-to/build-locally/", "https://docs.digitalocean.com/products/app-platform/how-to/destroy-app/"])
    loader.requests_per_second = 1
    docs = loader.load()  

    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 70)
    document_chunks = text_splitter.split_documents(docs)
    print(document_chunks)
    # Create a vector store from the chunks
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(document_chunks, embedding=embeddings)
    
    return vector_store

# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences minimum and keep the "
    "answer concise."
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