import glob
import json
import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    images_list = sorted(glob.glob("Radiographs/*.JPG"))
    segmentation_list = sorted(glob.glob("Segmentation/teeth_mask/*.jpg"))
    masks_list = sorted(glob.glob("Segmentation/maxillomandibular/*.jpg"))


    train_ids, val_ids = train_test_split(range(len(images_list)), test_size=0.3, shuffle=True, random_state=12345)
    valid_ids, test_ids = train_test_split(val_ids, test_size=0.333, shuffle=True, random_state=6789)

    out = {
        "class_names": ["background", "teeth"],
        "class_weights": {},
        "train": [],
        "valid": [],
        "test": []
    }

    for k in ["train", "valid", "test"]:
        for id in tqdm.tqdm(eval(f"{k}_ids")):
            out[k].append(
                {
                    "img": images_list[id],
                    "seg": segmentation_list[id],
                    "msk": masks_list[id],
                }
            )

    num_classes = len(out["class_names"])
    num_samples_class = np.zeros(num_classes)
    for data in tqdm.tqdm(out["train"], total=len(out["train"])):
        file = data["seg"]
        seg = np.asarray(Image.open(file).convert("1"), dtype="uint8")
        for c in range(num_classes):
            num_samples_class[c] += len(seg[seg == c])

    num_samples = seg.shape[0] * seg.shape[1] * len(out["train"])
    class_weights = num_samples / (num_classes * num_samples_class)

    for c in range(num_classes):
        out["class_weights"][c] = class_weights[c]

    json.dump(out, open("data.json", "w"), indent=4)
