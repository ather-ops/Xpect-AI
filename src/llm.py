# Dotenv
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
client=Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Generate query function
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are a movie-finder AI assistant.
Use only the provided movie context.
Do not invent movies or information.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}
User Query:
{query}
"""
            }
        ]
    )

    return response.choices[0].message.content
print("Done all works perfect")
