from langchain.document_loaders import TextLoader

loader = TextLoader("inputFile.txt")
loader.load()

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='inputRecord.csv')
data = loader.load()

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("inputDocument.pdf")
pages = loader.load_and_split()