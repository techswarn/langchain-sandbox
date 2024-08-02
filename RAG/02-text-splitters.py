from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # The chunk_size and chunk_overlap can be modified according to the requirements
    length_function = len,
    chunk_size = 200,
    chunk_overlap  = 10,
    add_start_index = True,
)

texts = text_splitter.create_documents([input_document])