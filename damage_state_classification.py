import numpy as np
from scipy.ndimage import label
import os
from PIL import Image
from scipy import ndimage
import argparse
import cv2
from tqdm import tqdm
from asko4mini import asko4mini

def get_parser():
    parser = argparse.ArgumentParser(description="Quake-OVUDA + VLM Hybrid Framework For Classifying Component Damage States")
    parser.add_argument(
        "--coloredinstances",
        help="Folder to output colored instance labels"
    )

    parser.add_argument(
        "--predictionsfolder",
        default="data/predictedClassLabels",
        help="Folder with Quake-OVUDA predicted component segmentations"
    )

    parser.add_argument(
        "--imagefolder",
        default="data/realbuildingDamageStates/images/val",
        help="Folder with real images to be tested and predicted on"
    )

    parser.add_argument(
        "--outputdir",
        default="data/output",
        help="Folder to output instance and prediction labels"
    )

    parser.add_argument(
        "--prompttype",
        default="short",
        help="Specified prompt type to be used by the VLM (short/detailed)"
    )


    return parser

def instances_from_component_preds(args):
    totalNumInstances = 0
    imagesCompleted = 0
    os.makedirs(os.path.join(args.outputdir, "instances"), exist_ok=True)
    if args.coloredinstances:
        os.makedirs(os.path.join(args.outputdir, args.coloredinstances), exist_ok=True)

    for img in os.listdir(args.predictionsfolder):

        labelImg = Image.open(os.path.join(args.predictionsfolder, img))
        labelArr = np.asarray(labelImg)

        instanceArr = np.zeros(labelArr.shape, dtype=np.uint16)
        totalInstanceCount = 1
        for classId in range(0, 5):
            if classId == 2 or classId == 3:
                continue
            idMask = labelArr == classId
            id_array, num_instances = label(idMask)
            

            for i in range(1, num_instances + 1):
                currIdMask = id_array == i
                currIdMask_filled = ndimage.binary_fill_holes(currIdMask)
                if np.count_nonzero(currIdMask_filled) < 40000:
                    continue
                instanceArr[currIdMask_filled] = totalInstanceCount
                totalInstanceCount += 1
                totalNumInstances += 1
        
        instancesImage = Image.fromarray(instanceArr)
        instancesImage.save(os.path.join(args.outputdir, "instances", img))

        if args.coloredinstances:
            instanceArrColored  = np.zeros((instanceArr.shape[0], instanceArr.shape[1], 3), dtype=np.uint8)
            for instanceID in np.unique(instanceArr):
                if instanceID == 0:
                    continue
                
                instIDMask = instanceArr == instanceID

                instanceArrColored[instIDMask] = list(np.random.choice(range(256), size=3))
            
            instancesImageColored = Image.fromarray(instanceArrColored)
            instancesImageColored.save(os.path.join(args.outputdir, args.coloredinstances, img))
        
        
        imagesCompleted += 1
        if imagesCompleted%5 == 0:
            print(f'Images completed: {imagesCompleted}')

def masked_component_classification(args):
    os.makedirs(os.path.join(args.outputdir, "vlmPredictedLabels"), exist_ok=True)
    os.makedirs(os.path.join(args.outputdir, "vlmPredictedLabels_colored"), exist_ok=True)

    for img in tqdm(os.listdir(args.imagefolder), desc="Grounding image components"):

        image = Image.open(os.path.join(args.imagefolder, img))
        image_rgba = image.convert("RGBA")
        image_arr = np.asarray(image_rgba)
        temp_image = image_arr.copy()
        
        imageInstances = Image.open(os.path.join(args.predictionsfolder, img))
        imageInstances_arr = np.asarray(imageInstances)
        instanceIds = np.unique_values(imageInstances_arr)

        for id in instanceIds:
            if id==0:
                continue
            temp_image = image_arr.copy()
            id_mask = imageInstances_arr == id

            temp_image[~id_mask] = [0,0,0,0]
            x, y, w, h, = cv2.boundingRect(temp_image[..., 3])
            im2 = temp_image[y:y+h, x:x+w, :]
            masked_component_img = cv2.cvtColor(im2, cv2.COLOR_RGBA2BGRA)


            temparr = np.ones(imageInstances_arr.shape, dtype=np.uint8) * 255

            outputClass = asko4mini(masked_component_img, args.prompttype)
            id_mask = imageInstances_arr == id
            temparr[id_mask] = outputClass



        Image.fromarray(temparr).save(os.path.join(args.outputdir, "vlmPredictedLabels", img))

        colorsDamStates = [[0,200,0], [250,250,0], [250,111,0], [250,0,0]]
        temparr_colored = np.zeros((temparr.shape[0], temparr.shape[1], 3), dtype=np.uint8)
        for lab in range(0, 4):
            labelMask = temparr == lab
            temparr_colored[labelMask] = colorsDamStates[lab]

        coloredLabelImg = Image.fromarray(temparr_colored)
        coloredLabelImg.save(os.path.join(args.outputdir, 'vlmPredictedLabels_colored'))




if __name__ == '__main__':
    args = get_parser().parse_args()
    os.makedirs(args.outputdir, exist_ok=True)

    instances_from_component_preds(args)
    masked_component_classification(args)