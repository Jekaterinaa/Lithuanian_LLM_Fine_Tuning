from ollama import chat

response = chat(
  model='gpt-oss',
  messages=[{'role': 'user', 'content': 'Tell me about Canada.'}],
  format='json'
)
print(response.message.content)