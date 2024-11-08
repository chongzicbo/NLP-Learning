import requests

url = "http://192.168.1.14:18001/uploadfile/"
files = {"file": open("/data/bocheng/data/docx/agriculture-2549214.docx", "rb")}
response = requests.post(url, files=files)

print(response.json())
