# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

# Modified by Charles Abdo
# - Include code for retrieving and outputting class labels

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from sed import add_sed_config
from predictor import VisualizationDemo

from PIL import Image
import numpy

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A directory to save output class labels." \
        "Will default to sed_labels_output in the current directory.",
    )

    parser.add_argument(
        "--outputcolored",
        help="A directory to save colored class labels."
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        if not args.output:
            args.output = "sed_label_output"
            os.makedirs(args.output, exist_ok=True)
        
        if args.outputcolored:
            os.makedirs(args.output, exist_ok=True)

        for pathName in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(pathName, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            predicted_labels = predictions["sem_seg"].detach().clone()
            
            pred_label_argmax = predicted_labels.argmax(dim=0).cpu()
            pred_label_numpy = pred_label_argmax.numpy()

            
            imgname = os.path.basename(pathName)

            pred_label_img = Image.fromarray((pred_label_numpy).astype(np.uint8))
            pred_label_img.save(os.path.join(args.output, imgname))

            color_map = {
                  0: (81,204,204),
                  1: (142,81,204),
                  2: (142,204,81),
                  3: (204,173,81),
                  4: (89,51,127),
            }

            # color_map = {
            #       0: (0,200,0),
            #       1: (250,250,0),
            #       2: (250,111,0),
            #       3: (250,0,0)
            # }            

            if args.outputcolored:
                imgheight, imgwidth = pred_label_numpy.shape

                colored_image = np.zeros((imgheight, imgwidth, 3), dtype=np.uint8)

                for class_id, color in color_map.items():
                    colored_image[pred_label_numpy == class_id] = color

                color_image_pil = Image.fromarray(colored_image)

                color_image_pil.save(os.path.join(args.outputcolored, imgname))

            logger.info(
                "{}: {} in {:.2f}s".format(
                    imgname,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
