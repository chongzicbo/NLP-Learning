###管道操作符
def add(a: str):
    return f"{a} ----- hello,world"


def first(text: str):
    return text


result = first | add

print(result("xiaoming"))
