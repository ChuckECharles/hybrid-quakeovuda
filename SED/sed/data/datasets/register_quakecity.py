import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

QUAKE_CATEGORIES = [{'color': [81,204,204], 'id': 1, 'name': 'wall'}, 
                   {'color': [142,81,204], 'id': 2, 'name': 'column'}, 
                   {'color': [142,204,81], 'id': 3, 'name': 'window'}, 
                   {'color': [204,173,81], 'id': 4, 'name': 'balcony'},
                   {'color': [89,51,127], 'id': 5, 'name': 'beam'}, 
                   ]

# QUAKE_CATEGORIES = [{'color': [0,200,0], 'id': 0, 'name': 'damagenone', 'trainId': 0}, 
#                    {'color': [250,250,0], 'id': 1, 'name': 'light', 'trainId': 1}, 
#                    {'color': [250,111,0], 'id': 2, 'name': 'moderate', 'trainId': 2}, 
#                    {'color': [250,0,0], 'id': 3, 'name': 'severe', 'trainId': 3},
#                    ]

def _get_quakecity_meta():
    stuff_ids = [k["id"] for k in QUAKE_CATEGORIES]
    assert len(stuff_ids) == 5, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in QUAKE_CATEGORIES]
    stuff_colors = [k["color"] for k in QUAKE_CATEGORIES]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_quakecity(root):
    rootDir = '/home/cgabdo/Documents/QuakeOVUDA/data'
    meta = _get_quakecity_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "quakecityComponents/images", "annotations_detectron2/train"),
        ("test", "realbuildingComponents/images/val", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(rootDir, image_dirname)
        gt_dir = os.path.join(rootDir, sem_seg_dirname)
        name = f"quakecity_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            # stuff_colors=meta["stuff_colors"],
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_quakecity(_root)
