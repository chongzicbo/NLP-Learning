import os
import sys
import argparse
from paddle.utils.download import get_path_from_url

URL = "https://bj.bcebos.com/paddlenlp/paddlenlp/datasets/waybill.tar.gz"


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="directory to save data to", type=str, default="./"
    )
    args = parser.parse_args(arguments)
    get_path_from_url(URL, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
