api_key = "9ckx4HLpVaFjJC4nf4TLepfRjbB8dEdF"

from mistralai import Mistral
import os
import base64
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, img_path in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({img_path})"
        )
    return markdown_str
# api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)


# uploaded_pdf = client.files.upload(
#     file={
#         "file_name": "uploaded_file.pdf",
#         "content": open(
#             "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/pdf/brainsci-2488662.pdf",
#             "rb",
#         ),
#     },
#     purpose="ocr",
# )
# print(uploaded_pdf.id)
uploaded_pdf_id = "06c50916-8f6a-4ba2-a3a1-372496bc2ca7"
client.files.retrieve(file_id=uploaded_pdf_id)
signed_url = client.files.get_signed_url(file_id=uploaded_pdf_id)
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    },
    include_image_base64=True,
)


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, img_path in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({img_path})"
        )
    return markdown_str


def save_ocr_results(ocr_response: OCRResponse, output_dir: str) -> None:
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_markdowns = []
    for page in ocr_response.pages:
        # 保存图片
        page_images = {}
        for img in page.images:
            img_data = base64.b64decode(img.image_base64.split(",")[1])
            img_path = os.path.join(images_dir, f"{img.id}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            page_images[img.id] = f"images/{img.id}.png"

        # 处理markdown内容
        page_markdown = replace_images_in_markdown(page.markdown, page_images)
        all_markdowns.append(page_markdown)

    # 保存完整markdown
    with open(os.path.join(output_dir, "complete.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))

if __name__ == "__main__":
    save_ocr_results(ocr_response, "ocr_results")