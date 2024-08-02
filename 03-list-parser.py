from langchain.output_parsers import CommaSeparatedListOutputParser
parser_List = CommaSeparatedListOutputParser()
partial_variables={"format_instructions": parser_List.get_format_instructions()},
print(llm.predict(text = prompt_List.format(question="ENTER_YOUR_QUERY_HERE")))