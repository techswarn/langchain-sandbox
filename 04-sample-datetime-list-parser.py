# importing LangChain modules
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser, CommaSeparatedListOutputParser


# Insert your key here
llm = OpenAI(openai_api_key = "{{OpenAI_Key}}")

parser_dateTime = DatetimeOutputParser()
parser_List = CommaSeparatedListOutputParser()

# creating our prompt template
template = """Provide the response in format {format_instructions} 
            to the user's question {question}"""

prompt_dateTime = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_dateTime.get_format_instructions()},
)

prompt_List = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_List.get_format_instructions()},
)

# formatting the output
print(llm.predict(text = prompt_dateTime.format(question="When was the first iPhone launched?")))
print(llm.predict(text = prompt_List.format(question="What are the four famous chocolate brands?")))