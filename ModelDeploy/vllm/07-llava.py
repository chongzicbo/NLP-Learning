import os

os.environ["XDG_CACHE_HOME"] = "/data/bocheng/data/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/bocheng/data/.cache/huggingface/hub/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from vllm import LLM
from vllm.assets.image import ImageAsset


def run_llava():
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    image = ImageAsset("stop_sign").pil_image

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_llava()
