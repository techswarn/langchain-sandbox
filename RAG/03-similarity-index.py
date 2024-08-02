from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load the document
input_document = TextLoader("./my_document.txt").load()

# transform the document
text_splitter = RecursiveCharacterTextSplitter(
    # The chunk_size and chunk_overlap can be modified according to the requirements
    length_function = len,
    chunk_size = 200,
    chunk_overlap  = 10,
    add_start_index = True,
)

documents = text_splitter.create_documents([input_document])

# embed the chunks
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# user query
query = "what are mudbanks?"

# computing the search using the similarity_search() method
docs = db.similarity_search(query)
#Using vector stores to compute the similarity
# embedding the query
embedding_vector = OpenAIEmbeddings().embed_query(query)

# computing the search using the search_by_vector() method
docs = db.similarity_search_by_vector(embedding_vector)

#Retrievers
db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

# invoking the retriever 
retrieved_docs = retriever.invoke(
    # write your query here
)