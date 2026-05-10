# memory_chatbot.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")
# Custome prompt
INSURANCE_TEMPLATE = """
You are an expert insurance advisor at InsureSafe India.
You help customers understand policies, claims, premiums, and coverage.
Always be helpful, clear, and professional.
Previous Conversation:
{history}
Customer: {input}
InsureSafe Advisor:"""
prompt = PromptTemplate(
input_variables=["history", "input"],
template=INSURANCE_TEMPLATE
)
llm = ChatOpenAI(
model="gpt-4o-mini",
temperature=0.4,
openai_api_key=os.getenv("OPENAI_API_KEY")
)
# Memory stores the rolling conversation
memory = ConversationBufferMemory(
memory_key="history",
return_messages=False
)
chain = ConversationChain(
llm=llm,
memory=memory,
prompt=prompt,
verbose=True # Shows full prompt including history
)
# Simulate a realistic multi-turn insurance conversation
conversation = [
"Hi, I want to buy health insurance for my family of 4.",
"What is the minimum sum insured I should go for?",
"Does it cover pre-existing conditions like diabetes?",
"How long is the waiting period for that?",
"Can you summarize what we discussed so far?",
]
for turn, user_msg in enumerate(conversation, 1):
    print(f"\n--- Turn {turn} ---")
    print(f"Customer: {user_msg}")
    response = chain.predict(input=user_msg)
    print(f"Advisor: {response}")
# View what is stored in memory
print("\n=== Memory Buffer ===")
print(memory.buffer)
# Clear memory (useful for starting new conversations)
memory.clear()
print("Memory cleared! Buffer:", memory.buffer)