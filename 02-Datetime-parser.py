from langchain.output_parsers import DatetimeOutputParser
parser_dateTime = DatetimeOutputParser()

# creating our prompt template
template = """Provide the response in format {format_instructions} 
            to the user's question {question}"""
            
# passing the format instructions of the parser to the template
prompt_dateTime = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_dateTime.get_format_instructions()},
)

print(llm.predict(text = prompt_dateTime.format(question="ENTER_YOUR_QUERY_HERE")))