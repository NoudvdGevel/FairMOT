# Models
This file will give an overview of the pre-trained models and the ones that will be created to track people in fisheye cameras.

## Networks
The two base networks provided by the repository are:

* [Deep Layer Aggregation Network](https://arxiv.org/pdf/1904.07850.pdf): Original adapted with more skip connections like in Feature Pyramid Network.
* [High-Resolution Network](https://arxiv.org/pdf/1902.09212.pdf)

HRNet achieve higher precision but DLANet reduces ID switches and increasees true positive rate.
More information about these networks can be found in their original papers and the [FairMOT](https://arxiv.org/pdf/2004.01888.pdf) paper.

## Pre-trained models
| Model    |  Base network | Dataset |
|--------------------|-----------|--------|
|ctdet_coco_dla_2x  | DLA-34 | COCO |
|hrnetv2_w32_imagenet_pretrained | HR-Net-w32|ImageNet|
|hrnetv2_w18_imagenet_pretrained | HR-Net-w32|ImageNet|

## Pre-trained tracking models
| Model    |  Base model | Dataset |
|--------------------|-----------|--------|
|all_dla34|ctdet_coco_dla_2x  | JDE |
|all_hrnet_v2_w18 | hrnetv2_w18_imagenet_pretrained |JDE|

### JDE dataset
The data used to train the tracking model is copied from the [JDE](https://arxiv.org/pdf/1909.12605v1.pdf) tracker. 
The data and storing structure can be found in their [Data ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md).
## Pre-trained tracking models
| Dataset    |  ETH | CP | CT | M16 | CS | PRW | TOTAL |
|------------|------|----|----|-----|----|-----|-------|
|Images|2K |3K |27K |    53K|11K|6K|54K|
|Boxes | 17K|    21K|    46K |   112K|  55K|18K|270K|
|IDs|-|-|0.6K|   0.5K|7K|0.5K|8.7K|

## Proposed models
The following model will be trained:
| Model    |  Base model | Dataset |
|--------------------|-----------|--------|
|overhead_dla34|ctdet_coco_dla_2x  | Fisheye |
|overhead_fine_dla34 | all_dla34 |Fisheye|
|overhead_rot_dla34 | overhead_dla34 |CEPDOF (rotation)|

### Fisheye dataset
| Dataset    |  CEPDOF | HABBOF | MWC | BOMNI | THEODORE | PETS2001-4 | HDA-cam2| Vicomtech-evaluation | Vicomtech-Bajas| TOTAL |
|------------|------|----|----|-----|----|-----|-------|-------|-------|-------|
|Images|25.5K |5.8K |...K | ...|   100K|11.8K|1.8K|5K?|50K?|...|
|Boxes | ...|...|...|...|...|...|...|...|...|...|
|IDs|Yes| No| Yes|Yes|Yes|Yes|Yes|Yes|Yes|-|
|Note| Rotated boxes  |Rotated boxes | Not finished| |Synthetic | | Overhead normal camera | Not sure if in other set| 5m and 20m camera. Inlcudes synthetic||

For training non-rotated detection model, the CEPDOF and HABBOF boxes will be rotated to 0 degrees.

### CEPDOF
The CEPDOF dataset includes rotated bounding boxes. This cannot be simultaneously trained with the other data. The authors report improved re-id accuracy using rotated boxes.
A pre-trained model can be fine-tuned with an additional loss function to achieve rotated bounding box detections.


