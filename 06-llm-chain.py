# importing the modules
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# defining the LLM model
llm = OpenAI(temperature=0.0, openai_api_key="{{OpenAI_Key}}")

# creating the prompt template
prompt_template = PromptTemplate(
    input_variables=["book"],
    template="Name the author of the book {book}?",
)

# creating the chain
chain = LLMChain(llm=llm, 
                prompt=prompt_template, 
                verbose=True)

# calling the chain
print(chain.run("The Da Vinci Code"))