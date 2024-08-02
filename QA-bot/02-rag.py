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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# For loading documents from web
from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio
nest_asyncio.apply()
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
#transformer pipeline 
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000)
llm = HuggingFacePipeline(pipeline=pipe)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)
# chain = prompt | llm.bind(skip_prompt=True)
# question = "Which is the bigest planet?"
# print(chain.invoke({"question": question}))

# # loading the document
# loader = PyPDFLoader("./doc.pdf")
# mypdf = loader.load() 


loader = WebBaseLoader(["https://docs.digitalocean.com/products/app-platform/", "https://docs.digitalocean.com/products/app-platform/getting-started/quickstart/"
                       "https://docs.digitalocean.com/products/app-platform/how-to/create-apps/", "https://docs.digitalocean.com/products/app-platform/how-to/deploy-from-container-images/", "https://docs.digitalocean.com/products/app-platform/how-to/deploy-from-monorepo/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/scale-app/", "https://docs.digitalocean.com/products/app-platform/how-to/add-deploy-do-button/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-components/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-services/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-jobs/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-workers/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/manage-static-sites/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-functions/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-data-storage/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/manage-databases/", "https://docs.digitalocean.com/products/app-platform/details/limits/"])
loader.requests_per_second = 1
docs = loader.load()

# Defining the splitter 
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20
)

# splitting the document
docs = document_splitter.split_documents(docs)

# embedding the chunks to vectorstores
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = 'db'

my_database = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

retriver = my_database.as_retriever()

instruction_to_system = """
Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is 
"""

question_maker_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_to_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

question_chain = question_maker_prompt | llm | StrOutputParser()
# value = question_chain.invoke({"question": "can you explain more?", "chat_history": [HumanMessage(content="you explained that the moon was round")]})
# print(value)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, provide a summary of the context. Do not generate your answer. \

{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]


retriever_chain = RunnablePassthrough.assign(
    context=contextualized_question | retriver
)

# retriever_chain.invoke({
#     "chat_history": [HumanMessage(content="you explained that the moon was round")],
#     "question": "can you explain more?"
# })


#RAG CHAIN

rag_chain = (
    retriever_chain | qa_prompt | llm
)

question = "How to scale app platform?"

chat_history = []

ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])
print(ai_msg)