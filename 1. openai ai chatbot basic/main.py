from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_model = os.environ["OPENAI_MODEL"]

client = OpenAI()

response = client.chat.completions.create(
    model=openai_model,
    max_tokens=50,
    n=1,
    temperature=0,
    messages=[
        {"role": "user", "content":"hello"}
    ]
)

print(response.choices[0].message.content)
