import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


information = """
 Jawad is a software engineer who loves to write code. He is a big fan of the football club Tottenham Hotspur FC.
 He loves to travel and explore new places. However, he is equally happy to spend time at home watching Netflix.
 He is a certified AWS Solutions Architect, Developer Associate and AI practitioner and has experience working with cloud technologies.
 He has a pet cat named Amina who he adores. He is a big foodie and loves to try out new cuisines.
"""

summary_template = """
    Given the information {information} about a person from I want you to create a very short summary and two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOpenAI(
    temperature=0, model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_API_KEY")
)

chain = summary_prompt_template | llm | StrOutputParser()

res = chain.invoke(input={"information": information})

print(res)
