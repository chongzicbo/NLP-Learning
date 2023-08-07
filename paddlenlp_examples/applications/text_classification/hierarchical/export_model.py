import argparse
import os

import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument(
    "--params_path",
    type=str,
    default="./checkpoint/",
    help="The path to model parameters to be loaded.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./export",
    help="The path of model parameter in " "static graph to be saved.",
)
args = parser.parse_args()

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ],
    )

    # Save in static graph model.
    save_path = os.path.join(args.output_path, "float32")
    paddle.jit.save(model, save_path)
