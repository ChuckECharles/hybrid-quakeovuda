import os
import os.path as osp
from pathlib import Path
import tqdm
from glob import glob

import numpy as np
from PIL import Image

QUAKE_CATEGORIES = [{'color': [81,204,204], 'id': 0, 'name': 'wall', 'trainId': 0}, 
                   {'color': [142,81,204], 'id': 1, 'name': 'column', 'trainId': 1}, 
                   {'color': [142,204,81], 'id': 2, 'name': 'window', 'trainId': 2}, 
                   {'color': [204,173,81], 'id': 3, 'name': 'balcony', 'trainId': 3},
                   {'color': [89,51,127], 'id': 4, 'name': 'beam', 'trainId': 4}, 
                   ]

# QUAKE_CATEGORIES = [{'color': [0,200,0], 'id': 0, 'name': 'damagenone', 'trainId': 0}, 
#                    {'color': [250,250,0], 'id': 1, 'name': 'light', 'trainId': 1}, 
#                    {'color': [250,111,0], 'id': 2, 'name': 'moderate', 'trainId': 2}, 
#                    {'color': [250,0,0], 'id': 3, 'name': 'severe', 'trainId': 3},
#                    ]


if __name__ == "__main__":
    # dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "quakecity"
    cwd = os.getcwd()
    target_dir = os.path.join(cwd, "..", "data")
    dataset_dir = os.path.abspath(target_dir)

    id_map = {}
    for cat in QUAKE_CATEGORIES:
        id_map[cat["id"]] = cat["trainId"]

    for name in ["train", "val"]:
        if name == "train":
            annotation_dir = f"{dataset_dir}/quakecityComponents/labels"
        if name == "val":
            annotation_dir = f"{dataset_dir}/realbuildingComponents/labels/val"

        output_dir = f"{dataset_dir}/annotations_detectron2/{name}"
        os.makedirs(output_dir, exist_ok=True)

        for file in tqdm.tqdm(os.listdir(annotation_dir)):
            # May need to change based on naming conventions of the used labels
            if "labelTrainId" not in file:
                continue
            output_file = f"{output_dir}/{file[:-18]}.png"
            # -----------------------------------------------------------------
            lab = np.asarray(Image.open(f"{annotation_dir}/{file}"))
            assert lab.dtype == np.uint8

            output = np.zeros_like(lab, dtype=np.uint8) + 255
            for obj_id in np.unique(lab):
                if obj_id in id_map:
                    output[lab == obj_id] = id_map[obj_id]

            Image.fromarray(output).save(output_file)