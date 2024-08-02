# importing the modules
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

# defining the LLM model 
llm = OpenAI(temperature=0.0, openai_api_key="{{OpenAI_Key}}")

# defining the original input 
biography = "He is an American author of thriller fiction, best known for his Robert Langdon series. \
          He has sold over 200 million copies of his books, which have been translated into 56 \
          languages. His other works include Angels & Demons, The Lost Symbol, Inferno, and Origin. \
          He is a New York Times best-selling author and has been awarded numerous awards for his \
          writing."

# creating the prompt template for the first chain
prompt_1 = ChatPromptTemplate.from_template(
    "Summarize this biography in one sentence:"
    "\n\n{biography}"
)

# we input the original biography, and the output here is the one-line biography
chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="one_line_biography")

# creating the prompt template for the second chain
prompt_2 = ChatPromptTemplate.from_template(
    "Can you tell the author's name in this biography:"
    "\n\n{one_line_biography}"
)

# we input the one-line biography, and the output here is the author's name
chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="author_name")

# creating the prompt template for the third chain
prompt_3 = ChatPromptTemplate.from_template(
    "Tell the name of the highest selling book of this author: "
    "\n\n{author_name}"
)

# we input the author's name, and the output here is the highest-selling book
chain_3 = LLMChain(llm=llm, prompt=prompt_3, output_key="book")

# creating the prompt template for the fourth chain
prompt_4 = ChatPromptTemplate.from_template(
   "Write a follow-up response to the following "
    "summary of the highest-selling book of the author:"
    "\n\nAuthor: {author_name}\n\nBook: {book}"
)

# we input the author's name and the highest-selling book, and the output is the book's summary
chain_4 = LLMChain(llm=llm, prompt=prompt_4, output_key="summary")

# combining the chains      
final_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["biography"],
    output_variables=["one_line_biography", "author_name","summary"],
    verbose=True
)

# calling the final chain
print(final_chain(biography))