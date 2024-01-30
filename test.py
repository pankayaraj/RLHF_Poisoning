# GPT
from openai import AzureOpenAI
import os
import logging
import time

def get_refined_caption( query):
      '''Refine coarse_caption according to refine_instruction'''
      
      api_call_success = False
      client = AzureOpenAI(
      azure_endpoint = "https://gpt1108.openai.azure.com/" ,#os.getenv("AZURE_OPENAI_ENDPOINT"), 
      api_key="cc011ce725e443b1b8345bf6e0186193", #os.getenv("AZURE_OPENAI_KEY"),  
      api_version="2023-05-15"
      )

      # print('Query to GPT is {}'.format(query))

      while not api_call_success:
            try:
                  response = client.chat.completions.create(
                  model="gpt-4", 
                  messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                  ]
                  )
                  api_call_success = True
            except BaseException:
                  logging.exception("An exception on GPT was thrown!")
                  print("Wait a while for GPT")
                  time.sleep(2)
      output = response.choices[0].message.content

      return output



print(get_refined_caption("What is a pen?"))








