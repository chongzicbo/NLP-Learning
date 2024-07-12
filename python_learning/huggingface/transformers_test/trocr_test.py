import os

os.environ["XDG_CACHE_HOME"] = "/data/bocheng/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/bocheng/.cache/huggingface/hub/"

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

image_path = "/home/bocheng/dev/source_code/nougat-latex-ocr/examples/test_data/eq1.png"
image = Image.open(image_path).convert("RGB")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values)
generated_ids = model.generate(pixel_values)
print(generated_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
