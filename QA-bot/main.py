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
        MessagesPlaceholder(variable_name="chat_history"),
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
                       "https://docs.digitalocean.com/products/app-platform/how-to/manage-databases/", "https://docs.digitalocean.com/products/app-platform/details/limits/", "https://docs.digitalocean.com/products/app-platform/how-to/build-run-commands/", "https://docs.digitalocean.com/products/app-platform/how-to/use-environment-variables/"
                       "https://docs.digitalocean.com/products/app-platform/how-to/cache-content/", "https://docs.digitalocean.com/products/app-platform/how-to/change-region/", "https://docs.digitalocean.com/products/app-platform/how-to/change-stack/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/upgrade-buildpacks/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-time-zone/", "https://docs.digitalocean.com/products/app-platform/how-to/add-ip-address/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/manage-domains/", "https://docs.digitalocean.com/products/app-platform/how-to/configure-cors-policies/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-internal-routing/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/url-rewrites/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-health-checks/", "https://docs.digitalocean.com/products/app-platform/how-to/view-logs/", "https://docs.digitalocean.com/products/app-platform/how-to/forward-logs/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/create-alerts/", "https://docs.digitalocean.com/products/app-platform/how-to/view-insights/",
                       "https://docs.digitalocean.com/products/app-platform/how-to/manage-deployments/", "https://docs.digitalocean.com/products/app-platform/how-to/update-app-spec/", "https://docs.digitalocean.com/products/app-platform/how-to/manage-source-repo/", "https://docs.digitalocean.com/products/app-platform/how-to/build-locally/", "https://docs.digitalocean.com/products/app-platform/how-to/destroy-app/"])
loader.requests_per_second = 1
docs = loader.load()

# Defining the splitter 
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 30
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

history_aware_retriever = create_history_aware_retriever(
    llm, my_database.as_retriever(search_kwargs={"k": 5}), condense_question_prompt
)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# # defining the loop for a conversation with the AI
while True:
    question = input("Enter your query: ")
    if question == 'exit': 
	    break 
    # getting the response
    result = convo_qa_chain.invoke(
        {
            "input": question,
            "chat_history": [],
        }
    )
    print(result['answer'])
