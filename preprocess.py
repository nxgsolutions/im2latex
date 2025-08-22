from os.path import join
import argparse
import pandas as pd

from PIL import Image
from torchvision import transforms
import torch


def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    images_dir = join(data_dir, "formula_images_processed")

    # Load CSV (instead of .lst)
    split_file = join(data_dir, f"im2latex_{split}_filter.csv")
    df = pd.read_csv(split_file)  # expects columns like: image_name, formula
    print("df",df)
    pairs = []
    transform = transforms.ToTensor()

    for _, row in df.iterrows():
        img_name = row["image"]
        formula = row["formula"]

        img_path = join(images_dir, img_name)
        img = Image.open(img_path)
        img_tensor = transform(img)

        pair = (img_tensor, formula)
        pairs.append(pair)

    # sort by image size
    pairs.sort(key=img_size)

    out_file = join(data_dir, f"{split}.pkl")
    torch.save(pairs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


def img_size(pair):
    img, formula = pair
    return tuple(img.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    splits = ["validate", "test", "train"]
    for s in splits:
        preprocess(args.data_path, s)
