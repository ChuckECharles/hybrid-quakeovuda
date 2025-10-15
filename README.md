# Integrating Generalist and Specialist AI Inspectors: A Hybrid Vision-Language Framework for Annotation-Free Post-Earthquake Component and Damage Segmentation


## Introduction
- We introduce Quake-OVUDA, a specialist domain adaptive model trained on synthetic data, which integrates open-vocabulary (OV) semantic segmentation to guide unsupervised domain adaptation (UDA). Specifically, we fine-tune the Simple Encoder-Decoder (SED) OV model on the synthetic QuakeCity dataset and a dictionary of component- and damage-related terms. This trained model is leveraged to generate pseudo-labels for real-world images that are used in the self-supervision process of the UDA model, DAFormer.
- To further improve performance on nuanced damage classes, we propose a framework combining this specialist model with vision-language models (VLMs), leveraging their general expertise and contextual reasoning to refine predictions and improve classification accuracy. Component classifications from the trained Quake-OVUDA specialist are used as input for the generalist VLMs, enabling them to provide refined and reasoned damage state predictions for individual components.
- We demonstrate that this hybrid framework outperforms direct segmentation, open-vocabulary segmentation, and baseline UDA methods, particularly for underrepresented or context-sensitive components and damage states. 
- We provide an effective integration of specialist UDA and OV segmentation models and generalist VLMs, laying a foundation for reduced dependence on manually annotated real-world data through further tuning and refinement of the proposed strategies.

## Installation
```
git clone https://github.com/ChuckECharles/hybrid-quakeovuda.git
cd hybrid-quakeovuda
conda create -n hybridquakeovuda python=3.8.5
conda activate hybridquakeovuda
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7
cd SED/open_clip/
make install
```


## Dataset Preparation
The displayed folder structures are shown with respect to the component and damage state segmentation tasks evaluated in this work. If evaluating/training is being performed for only one of the two explored tasks, only the data for that task is required. The following folder structure is functional for the evaluation and training scripts described later:

```none
DAFormer
SED
data
├──quakecityComponents
|   ├──images
|   ├──labels (QuakeCity labels for the component seg. task)
├──quakeCityDamageStates
|   ├──images
|   ├──labels (QuakeCity labels for the damage state seg. task)
├──realbuildingComponents
|   ├──images
|   |  ├──train
|   |  ├──val
|   ├──labels
|   |  ├──train (Where SED pseudo-labels are saved. Not needed for DAFormer training)
|   |  ├──val
├──realbuildingDamageStates
|   ├──images
|   |  ├──train
|   |  ├──val
|   ├──labels
|   |  ├──train (Where SED pseudo-labels are saved. Not needed for DAFormer training)
|   |  ├──val
├──...
```

Data preprocessing scripts must be run for DAFormer and SED for both evaluation and training:
```shell
cd DAFormer
python tools/convert_datasets/quakecityComponents.py ../data/quakecityComponents --nproc 8
python tools/convert_datasets/quakecityDamageStates.py ../data/quakecityDamageStates --nproc 8
python tools/convert_datasets/realbuildingComponents.py ../data/realbuildingComponents --nproc 8
python tools/convert_datasets/realbuildingDamageStates.py ../data/realbuildingDamageStates --nproc 8

cd ../SED
python datasets/prepare_quakecity.py
# This will create an additional data directory in the data root folder for SED to use
```
## Segmentation and Hybrid Framework Evaluation 
This section will layout how evaluation can be preformed to retrieve performance metrics for Quake-OVUDA, DAFormer, and SED segmentation. 

Additionally, this section will cover how output from Quake-OVUDA and DAFormer can be passed to VLMs to evaluate the hybrid framework performance in classifying component damage states.

### Quake-OVUDA and DAFormer
Checkpoints for Quake-OVUDA and DAFormer for each segmentation task can be downloaded below:
<div style="text-align: center;">
  <table style="margin: 0 auto;">
    <tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Task</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr>
<td align="center">Quake-OVUDA</a></td>
<td align="center">Component Types</td>
<td align="center">43.24</td>
<td align="center"><a href="https://drive.google.com/file/d/1haIz96EG9Nq7nRRRug5Ht-tZ9HWf_i-l/view?usp=sharing">ckpt</a>&nbsp;
</tr>

<tr>
<td align="center">DAFormer</a></td>
<td align="center">Component Type</td>
<td align="center">37.75</td>
<td align="center"><a href="https://drive.google.com/file/d/1WQ1oAzt0S2HGO6edX80vDnKoVsPWpsyM/view?usp=sharing">ckpt</a>&nbsp;
</tr>

<tr>
<td align="center">Quake-OVUDA</a></td>
<td align="center">Damage States</td>
<td align="center">25.74</td>
<td align="center"><a href="https://drive.google.com/file/d/1IelSN7E6sIIkd8JSfzJdtfBWFxBQDDRq/view?usp=sharing">ckpt</a>&nbsp;
</tr>

<tr>
<td align="center">DAFormer</a></td>
<td align="center">Damage States</td>
<td align="center">23.48</td>
<td align="center"><a href="https://drive.google.com/file/d/1jkW6gtVhJEUXqS-Ji3gJKUTdK3iZGjZi/view?usp=sharing">ckpt</a>&nbsp;
</tr>
    </tbody>
  </table>
</div>

Place these in the 'DAFormer' directory.

In the DAFormer directory, run the following to get IoU segmentation metrics for each task:
```shell
# For Quake-OVUDA Component Type Segmentation
python -m tools.test configs/daformer/QuakeOVUDA_componentClassification.py path/to/task/weights.pth --eval mIoU

# For Quake-OVUDA Damage State Segmentation
python -m tools.test configs/daformer/QuakeOVUDA_damageStateClassification.py path/to/task/weights.pth--eval mIoU

# For DAFormer Component Type Segmentation
python -m tools.test configs/daformer/DAFormer_componentClassification.py path/to/task/weights.pth --eval mIoU

# For DAFormer Damage State Segmentation
python -m tools.test configs/daformer/DAFormer_damageStateClassification.py path/to/task/weights.pth --eval mIoU
```

### Hybrid Framework/VLM Damage State Classification
To pass Quake-OVUDA component predictions for evaluation by VLMs, first generate Quake-OVUDA predictios on the test set of real images for damage state segmentation:
```shell
# In the DAFormer directory

python -m tools.test configs/daformer/QuakeOVUDA_componentInferenceOnDamStateTestSet.py path/to/Quake-OVUDA_ComponentSegWeights.pth --eval mIoU
```
These will create a directory named 'predictedClassLabels' in the data root directory (make sure this has not already been created using previous DAFormer evaluation scripts). This folder contains predicted labels for each test set image with per-pixel classifications. These will be used to format the real-world test set images so that they can be passed to VLMs for damage state classification.


The following script will use the predicted component labels to create individual component instances. For each image, its instances are cropped and given to a VLM for prediction (only a script for prompting GPT-o4mini is provided (with API key removed), as Gemini VLMs require system login to use). These predictions are then used to create grounded prediction labels saved in the output directory.
```shell
python damage_state_classification.py

# To evaluate predictions, run the mIoU calculation script
python miou_eval.py
```

### SED

Weights for SED tuned using QuakeCity can be downloaded below for each task.

<div style="text-align: center;">
  <table style="margin: 0 auto;">
    <tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Task</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr>
<td align="center">Component Types</td>
<td align="center"><a href="">ckpt</a>&nbsp;
</tr>

<tr>
<td align="center">Damage States</td>
<td align="center"><a href="">ckpt</a>&nbsp;
</tr>
    </tbody>
  </table>
</div>

To get IoU segmentation metrics, run the following.
```shell
sh eval.sh configs/convnextL_768.yaml 1 output/ MODEL.WEIGHTS path/to/task/weights.pth
```

## Training
This section will describe how to train SED and DAFormer and how SED psuedo-labels can be created to train Quake-OVUDA.


### SED
To train SED, first download a pretrained checkpoint from the <a href="https://github.com/xb534/SED">SED GitHub</a>.

Run the following script to fine-tune SED on the QuakeCity synthetic  dataset:
```shell
# Component Type Segmentation Task
sh run.sh configs/convnextL_768.yaml 1 output/ MODEL.WEIGHTS path/to/weights.pth MODELS.SEM_SEG_HEAD.TRAIN_CLASS_JSON datasets/quakecitycomponents.json MODELS.SEM_SEG_HEAD.TEST_CLASS_JSON datasets/quakecitycomponents.json

# Damage State Segmentation Task
sh run.sh configs/convnextL_768.yaml 1 output/ MODEL.WEIGHTS path/to/weights.pth MODELS.SEM_SEG_HEAD.TRAIN_CLASS_JSON datasets/quakecitydamagestates.json MODELS.SEM_SEG_HEAD.TEST_CLASS_JSON datasets/quakecitydamagestates.json
```

To retrieve pseudo-labels from SED to train Quake-OVUDA, run the following script:
```shell
# For Component Type Segmentation
python demo/demo_for_vis.py --config-file configs/convnextL_768.yaml --input ../data/realbuildingComponents/images/val/* --output ../data/realbuildingComponents/labels/train

# For Damage State Segmentation
python demo/demo_for_vis.py --config-file configs/convnextL_768.yaml --input ../data/realbuildingDamageStates/images/val/* --output ../data/realbuildingDamageStates/labels/train
```
These will create psuedo-labels from the real, train set of images and place them in the corresponding folder as explained in the data processing section.

### Quake-OVUDA
First, download SegFormer-B5 weights from the <a href="https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training">SegFormer GitHub</a> and place them in the `/pretrained` folder in the DAFormer directory.

To train Quake-OVUDA using SED psuedo-labels, run the following (data processing scripts may need to be re-run in order to get the pseudo-labels in the correct format):

```shell
# Component Type Segmentation
python run_experiments --config DAFormer/configs/daformer/QuakeOVUDA_componentClassification.py

# Damage State Segmentation
python run_experiments --config DAFormer/configs/daformer/QuakeOVUDA_damageStateClassification.py
```

This will create a directory in the `work_dirs` directory where logs and training checkpoints will be stored.


### DAFormer
To train base DAFormer, run the following:

 ```shell
# Component Type Segmentation
python run_experiments --config DAFormer/configs/daformer/DAFormer_componentClassification.py

# Damage State Segmentation
python run_experiments --config DAFormer/configs/daformer/DAFormer_damageStateClassification.py
```

This will create a directory in the `work_dirs` directory where logs and training checkpoints will be stored.