"""
HumanCellAI - OpenAI API Connection Test
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-5.6")

print("=" * 70)
print("HumanCellAI - OpenAI API Test")
print("=" * 70)

if api_key is None:
    print("\nERROR: OPENAI_API_KEY not found in .env file.")
    exit()

print("Model:", model_name)

try:
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        input="Explain single-cell RNA sequencing in one short paragraph."
    )

    print("\n")
    print("=" * 70)
    print("API Connection Successful")
    print("=" * 70)
    print("\nResponse:\n")

    print(response.output_text)

except Exception as e:
    print("\nERROR:")
    print(e)