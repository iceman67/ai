import requests;

url = "http://localhost:8000/svm"
payload = {
	'exp': [2, 4]
}

r = requests.post(url,json=payload)
print(int (r.json()))
