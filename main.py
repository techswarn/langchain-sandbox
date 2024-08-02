# importing LangChain modules
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initiating the chat model with API key
llm = OpenAI(openai_api_key = "{{OpenAI_Key}}")

email_template = PromptTemplate.from_template(
    "Create an invitation email to the recipinet that is {recipient_name} \
 for an event that is {event_type} in a language that is {language} \
 Mention the event location that is {event_location} \
 and event date that is {event_date}. \
 Also write few sentences about the event description that is {event_description} \
 in style that is {style} "
)

message = email_template.format(
    style = "enthusiastic tone",
    language = "American english",
    recipient_name="John",
    event_type="product launch",
    event_date="January 15, 2024",
    event_location="Grand Ballroom, City Center Hotel",
    event_description="an exciting unveiling of our latest innovations"
    )

response = llm.invoke(message)
print(response)


