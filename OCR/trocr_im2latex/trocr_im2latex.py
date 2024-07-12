import os

os.environ["XDG_CACHE_HOME"] = "/data/bocheng/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/bocheng/.cache/huggingface/hub/"
from transformers import TrOCRProcessor
import glob
from tqdm import tqdm


from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import torch
import random


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
        self.images = [path for path in glob.glob(join(root_dir, "*.png"))][:1]
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
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            text, img_path = self.pairs[random.randint(0, len(self.pairs) - 1)]
            image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding


from transformers import TrOCRProcessor

root_dir = "/data/bocheng/data/MathOCR/nougat_latex"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
train_dataset = Im2Latex100k(
    root_dir=os.path.join(root_dir, "train"),
    formula_path=os.path.join(root_dir, "math.txt"),
    processor=processor,
)
val_dataset = Im2Latex100k(
    root_dir=os.path.join(root_dir, "val"),
    formula_path=os.path.join(root_dir, "math.txt"),
    processor=processor,
)
train_batch_size = 16
eval_batch_size = 16
version = 5
report_step = 100
num_epochs = 10
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=10, shuffle=True
)
eval_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=10)
from transformers import VisionEncoderDecoderModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
model.to(device)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = processor.tokenizer.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 800  # 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.tokenizer = processor.tokenizer
from datasets import load_metric


cer_metric = load_metric("cer")


def computer_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer


def evaluate(model, val_dataloader):
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            outputs = model.generate(batch["pixel_values"].to(device))
            cer = computer_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer
    print("validation CER:", valid_cer / len(val_dataloader))


from transformers import AdamW
from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    # train
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs
        for k, v in batch.items():
            # print(v)
            batch[k] = v.to(device)

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if i % report_step == 0:
            print(f"Train Loss: {loss.item()}")
    print(f"Train Loss after epoch {epoch}:", train_loss / len(train_dataloader))
    evaluate(model, eval_dataloader)
    model.save_pretrained(f"version_{version}/epoch_{epoch}")


model.save_pretrained(f"version_{version}/final")
