import requests

url = "http://127.0.0.1:8000/query"
data = {"question": "What is cancer?"}

response = requests.post(url, json=data)
print(response.json())  # Should return the chatbot's answer
