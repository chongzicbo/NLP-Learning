import httpx

# GET请求（带查询参数）
params = {"page": 1, "size": 20}
response = httpx.get("https://httpbin.org/get", params=params)
print(response.json())  # 查看解析后的JSON数据

# POST请求（多种数据格式）
data = {"key": "value"}
response = httpx.post("https://httpbin.org/post", json=data)  # JSON编码
# 或者使用form数据
response = httpx.post("https://httpbin.org/post", data=data)  # Form编码
# 或者发送原始数据
response = httpx.post("https://httpbin.org/post", content=b"raw_data")

response = httpx.get("https://httpbin.org/get")

# 常用属性
print(f"状态码: {response.status_code}")
print(f"响应头: {response.headers}")
print(f"编码: {response.encoding}")

# 内容获取方式
text_content = response.text  # 自动解码文本
binary_content = response.content  # 原始字节
json_content = response.json()  # JSON解析（可能抛出异常）

# 流式读取（适合大响应）
with httpx.stream("GET", "https://example.com/large_file") as response:
    for chunk in response.iter_bytes():
        process_chunk(chunk)
# 