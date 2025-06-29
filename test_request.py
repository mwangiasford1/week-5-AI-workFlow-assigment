import requests

data = {"feature1": 25, "feature2": 5.9, "feature3": 1}
res = requests.post("http://127.0.0.1:5000/predict", json=data)
print(res.status_code)
print(res.json())
