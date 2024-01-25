import os
import openai
from openai import AzureOpenAI

"""
openai.api_type = "azure"
openai.api_base = "https://gpt1108.openai.azure.com/" 
openai.api_key = "cc011ce725e443b1b8345bf6e0186193"
openai.api_version = "2023-05-15"
"""

client = AzureOpenAI(
  api_key = "https://gpt1108.openai.azure.com/" ,  
  api_version = "2023-05-15",
  azure_endpoint = "cc011ce725e443b1b8345bf6e0186193"
)
response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
)
print(response)
