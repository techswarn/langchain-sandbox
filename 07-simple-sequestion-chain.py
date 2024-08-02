# importing the modules
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# defining the LLM model for the first chain
llm = OpenAI(temperature=0.0, openai_api_key="{{OpenAI_Key}}")

# creating the prompt template and the first chain
prompt_1 = PromptTemplate(
    input_variables=["book"],
    template="Name the author who wrote the book {book}?"
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

# creating the prompt template and the second chain
prompt_2 = PromptTemplate(
    input_variables=["author_name"],
    template="Write a 50-word biography for the following author:{author_name}"
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

# combining the chains into a simple sequential chain
simple_sequential_chain = SimpleSequentialChain(chains=[chain_1, chain_2],verbose=True)

# running the simple sequential chain                                            
simple_sequential_chain.run("The Da Vinci Code")