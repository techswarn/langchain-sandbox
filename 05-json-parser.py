from typing import List
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator

# insert your key here
model = OpenAI(openai_api_key = "{{OpenAI_Key}}")

# defining the author class for the model
class Author(BaseModel):
    number: int = Field(description="number of books written by the author")
    books: List[str] = Field(description="list of books they wrote")

user_query = "Generate the books written by Dan Brown."

# defining the output parser
output_parser = PydanticOutputParser(pydantic_object=Author)

# defining the prompt template with parser instructions
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# defining the prompt
my_prompt = prompt.format_prompt(query=user_query)

# coverting the output to a string while generating the response
output = model(my_prompt.to_string())

# printing the result
print(output_parser.parse(output))