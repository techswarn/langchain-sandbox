from langchain_huggingface.llms import HuggingFacePipeline
#https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_pipelines/
hf = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it", task="text-generation", pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256},
)


from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))