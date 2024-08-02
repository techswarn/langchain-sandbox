# importing the modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# For loading documents from web
from langchain_community.document_loaders import WebBaseLoader

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


loader = WebBaseLoader("https://docs.digitalocean.com/products/app-platform/how-to/manage-databases/")
docs = loader.load()

# Defining the splitter 
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# splitting the document
docs = document_splitter.split_documents(docs)

# embedding the chunks to vectorstores
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'db'

my_database = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# # defining the conversational memory
retaining_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# defining the retriever

condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

qa_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. Use three sentences maximum and keep the
answer concise.

Chat History:
{chat_history}

Other context:
{context}

Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
question_answering = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=my_database.as_retriever(),
    memory=retaining_memory,
    condense_question_prompt=condense_question_prompt,
    combine_docs_chain_kwargs={
        "prompt": qa_prompt,
    },
)

# defining the loop for a conversation with the AI
while True:
    question = input("Enter your query: ")
    if question == 'exit': 
	    break 
    # getting the response
    result = question_answering({"question": "Answer only in the context of the document provided." + question})
    print(result['answer'])