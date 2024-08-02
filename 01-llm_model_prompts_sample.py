# importing LangChain modules
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# Initiating the chat model with API key
chat = ChatOpenAI(temperature=0.0, openai_api_key = "{{OpenAI_Key}}")

examples = [
  {
    "review": "I absolutely love this product! It exceeded my expectations.",
    "sentiment": "Positive"
  },
  {
    "review": "I'm really disappointed with the quality of this item. It didn't meet my needs.",
    "sentiment": "Negative"
  },
  {
    "review": "The product is okay, but there's room for improvement.",
    "sentiment": "Neutral"
  }
]

example_prompt = PromptTemplate(
                        input_variables=["review", "sentiment"], 
                        template="Review: {review}\n{sentiment}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Review: {input}",
    input_variables=["input"]
)

message = prompt.format(input="The machine worked okay without much trouble.")

response = chat.invoke(message)
print(response.content)

# importing LangChain modules
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# Initiating the chat model with API key
chat = ChatOpenAI(temperature=0.0, openai_api_key = "{{OpenAI_Key}}")

examples = [
  {
    "review": "I absolutely love this product! It exceeded my expectations.",
    "sentiment": "Positive"
  },
  {
    "review": "I'm really disappointed with the quality of this item. It didn't meet my needs.",
    "sentiment": "Negative"
  },
  {
    "review": "The product is okay, but there's room for improvement.",
    "sentiment": "Neutral"
  }
]

example_prompt = PromptTemplate(
                        input_variables=["review", "sentiment"], 
                        template="Review: {review}\n{sentiment}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Review: {input}",
    input_variables=["input"]
)

message = prompt.format(input="The machine worked okay without much trouble.")

response = chat.invoke(message)
print(response.content)