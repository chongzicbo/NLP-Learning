import requests
import os
from xml.dom import minidom
from time import perf_counter
from functools import wraps
from typing import List


def timeit(loop: int = 1):
    """
    函数执行失败时，重试

    :param loop: 循环执行次数
    :return:
    """

    # 校验参数，参数值不正确时使用默认参数
    if loop < 1:
        loop = 1

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sum_time: float = 0.0
            for i in range(loop):
                start_time: float = perf_counter()
                ret = func(*args, **kwargs)
                end_time: float = perf_counter()
                sum_time += end_time - start_time

            print(
                f"函数({func.__name__})共执行{loop}次，平均执行时间 {(sum_time/loop):.3f} 秒"
            )
            return ret

        return wrapper

    return decorator


pdf_dir = "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/pdf"
# output_dir = "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/output"
# url = "http://localhost:8070/api/processReferences"
# for file in os.listdir(pdf_dir):
#     file_path = os.path.join(pdf_dir, file)
#     # 文件路径
#     filename = file_path.split("/")[-1]
#     # 打开文件并准备作为表单数据的一部分发送
#     with open(file_path, "rb") as file:
#         # 构建表单数据字典
#         files = {
#             "input": (filename, file),  # 使用文件名作为键值对中的键
#         }
#         data = {"includeRawCitations": "1"}  # 注意这里需要转换成字符串形式

#         # 发送POST请求
#         response = requests.post(url, files=files, data=data)

#     # 检查响应状态码
#     if response.status_code == 200:
#         print(f"成功:{filename}")
#         with open(os.path.join(output_dir,filename.split(".")[0] + ".xml"), "w") as f:
#             f.write(response.text)
#     else:
#         print(f"失败: {response.status_code}, {response.text}")


class GrobidReferenceExtractor:
    def __init__(self, url):
        self.url = url

    @timeit(loop=1)
    def extract_references(self, file_path: str) -> List[str]:
        filename = file_path.split("/")[-1]
        extracted_references = []
        with open(file_path, "rb") as file:
            files = {
                "input": (filename, file),
            }
            data = {"includeRawCitations": "1"}

            response = requests.post(self.url, files=files, data=data)

            if response.status_code == 200:
                xml_res = response.text
                document = minidom.parseString(xml_res)
                references = document.getElementsByTagName("note")
                for reference in references:
                    if reference.getAttribute("type") == "raw_reference":
                        extracted_references.append(reference.firstChild.data)
            else:
                print(f"失败: {response.status_code}, {response.text}")
        return extracted_references


if __name__ == "__main__":
    url = "http://localhost:8070/api/processReferences"
    filepath = "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/pdf/cells-2489525.pdf"
    extractor = GrobidReferenceExtractor(url)
    # references = extractor.extract_references(filepath)
    # print(f"references number: {len(references)}")
    for file in os.listdir(pdf_dir):
        file_path = os.path.join(pdf_dir, file)
        # 文件路径
        filename = file_path.split("/")[-1]
        extracted_references=extractor.extract_references(file_path)
        print(f"{filename}: {len(extracted_references)}")
