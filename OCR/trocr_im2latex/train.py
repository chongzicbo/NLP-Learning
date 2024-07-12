import os

os.environ["XDG_CACHE_HOME"] = "/data/bocheng/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/bocheng/.cache/huggingface/hub/"
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import glob
from tqdm import tqdm

from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")


class Im2Latex100k(Dataset):
    def __init__(
        self,
        root_dir: str,
        formula_path: str,
        processor,
        max_target_length=500,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.max_target_length = max_target_length
        self.images = [path for path in glob.glob(join(root_dir, "*.png"))]
        eqs = open(formula_path, "r").read().split("\n")
        self.indices = [int(os.path.basename(img).split(".")[0]) for img in self.images]

        self.pairs = list()
        for i, im in tqdm(enumerate(self.images), total=len(self.images)):
            self.pairs.append((eqs[self.indices[i]], im))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        batch = self.pairs[index]
        text, img_path = batch
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensor="pt").pixel_values
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids
        labels = [
            label if label != self.tokenizer.pad_token_id else -100 for label in labels
        ]
        encoding = {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
        return encoding


if __name__ == "__main__":
    from transformers import TrOCRProcessor

    root_dir = "/data/bocheng/data/MathOCR/nougat_latex"
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    from transformers import VisionEncoderDecoderConfig

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    config = VisionEncoderDecoderConfig.from_pretrained("microsoft/trocr-small-printed")
    print(config.encoder)
