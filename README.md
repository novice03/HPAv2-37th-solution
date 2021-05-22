# Kaggle Human Protein Atlas - Single Cell Classification Challenge 37th place solution 

## Competition overview

The aim of the [Human Protein Atlas - Single Cell Classification Challenge](https://www.kaggle.com/c/hpa-single-cell-image-classification) hosted on Kaggle was to build models that recognize protein localization patterns in single cells. The training and test data consisted of slide-level images, each consisting of multiple cells. The challenging part of the competition was that the training data consisted of only image-level labels, which implies that all of the cells in an image may not have the same label as their slide-level labels. Therefore, this was a weakly supervised multi-label classification problem.

## Team information

Team name: the one true novice

Team members: novice03  (solo)

## Final submissions

1. 2 cell-level models and 1 slide-level model - Private 0.45904
2. 3 cell-level models - Private 0.45868

Final standing: 37/757 (top 4.8%)

## Solution overview

At a high level, my best submission (private LB score 0.45904) is an ensemble of 2 CNNs trained on cell-level images and 1 CNN trained on slide-level images, and my second-best submission (private LB score 0.45868) is an ensemble of 3 CNNs trained on a dataset of approximately 180,000 images of single cells. The most important part of my solution is my dataset preprocessing method, which allowed me to train my models on a relatively small dataset without sacrificing accuracy and, most importantly, tackled the weak label problem by reducing noise in the cell-level dataset by removing mis-labelled images. Although the score is not outstanding when compared to other top competitors, I feel it is a good score given the nature of the models (2 or 3 models trained only on cell-level images without the use of techniques such as pseudo-labelling, GAP, CAMs, Gridify, etc) and the size of the dataset used. More details about the data preparation are covered below.

## Data preparation

Two models in the first submission and all three models in the second submisison were trained only on cell-level images (images of single cells) extracted from slide-level images using HPACellSeg (un-modified). Public HPA data was used along with the images provided in the competition. All cell-level images were given the same labels as their slide-level image. For example, if a raw slide-level image labelled ‘0|1’ had 5 cells, all 5 cell-level images were given the label ‘0|1’. The only major problem is that the dataset of cell-level images is very noisy since the labels inherited from slide-level image were weak labels. Mis-labelled cell-level images worsen the model’s performance. So, to reduce the amount of noise in the dataset, (non-deep learning) heuristics were developed for each class that gave a fair indication of whether a given image belonged to that class. These heuristics were used to remove mis-labelled cell-level images and select a subset of the cell-level image dataset. This subset contains a much greater proportion of correctly labelled images compared to the whole set of cell-level images. Some specific heuristics are discussed in the table below:

Class ID|Class|Heuristic (a correctly labelled image of class x would have)
|-------|-----|---------|
0|Nucleoplasm|High average of pixel intensities in the green channel in the location of nucleus (found through nucleus mask from HPACellSeg)
1|Nuclear Membrane|High average of pixel intensities in the green channel in the border of the nucleus.
2|Nucleoli|High average of pixel intensities in the green channel in the nucleoli, which appear as darker spots in the nucleus (locations can be detected with a fairly high accuracy using simple thresholding).
3|Nucleoli fibrillar center|Same as (2)
4|Nuclear speckles|Same as (0), but a lower average
5|Nuclear bodies|Same as (4)
6|Endoplasmic reticulum|High similarity between green channel and yellow channel. PHash algorithm was used to find the hamming distance between the two channels. Images with the lowest hamming distance are more likely to be correctly labelled.
7|Golgi apparatus|Moderately high sum of pixel intensities in the green channel in the cytosol, whose location is given by cell mask XOR nucleus mask
10|Microtubules|Same as (6), but with red channel
13|Plasma membrane| High average of pixel intensities in the green channel in the location of the cytosol
14|Mitochondria|High average of pixel intensities in the green channel along the microtubules (used the red channel as a binary mask)
16|Cytosol|Same as (13)

For two classes, Vesicles and punctate cytosolic patterns and Mitotic Spindle, I relied on manual inspection to remove incorrectly labelled images. For the rest of the classes, I used the above heuristics to filter out incorrectly labelled images. No attempt was made to change the labels of images since these heuristics are not perfect, nor are they definitive of an image's true label. 

An example with pseudocode for the nucleoplasm class is as follows:

```
nuc_g_sum = 0 # sum of pixel intensities in the green channel at the nucleus
nuc_coords = nucleus.index # list of co-ordinates occupied by the nucleus

for coord in nuc_coords:
  nuc_g_sum += green_channel[coord]
 
nuc_g_avg = nuc_g_sum / len(nuc_coords)

if nuc_g_avg is high:
  keep image
 else:
  remove image

```
A more complex version of the above algorithm was repeated for multiple images with '0' in their label. A similar process was repeated for all images to filter out images that were incorrectly labelled. I found that, on average, about 30-50% of the images were removed per unique label. Below are some specific examples in which the heuristics successfully identify mis-labelled images:

Image ID|Label|Image|High average of pixel intensities in the green channel in the nucleoli?|High average of pixel intensities in the green channel in the the cytosol?|Keep image?
|--------|-----|-----|----------------------------------------------------------------------|----------------------------------------------------------------------|------------|
8c683d82-bbaa-11e8-b2ba-ac1f6b6435d0_9|2, 16| ![8c683d82-bbaa-11e8-b2ba-ac1f6b6435d0_9](https://i.imgur.com/4j5F5Hm.png)|Yes|Yes|Yes
bc16fab0-bb9f-11e8-b2b9-ac1f6b6435d0_15|2, 16| ![bc16fab0-bb9f-11e8-b2b9-ac1f6b6435d0_15](https://i.imgur.com/fx1Fm2h.png)|No|Yes|No
27e2a860-bbc9-11e8-b2bc-ac1f6b6435d0_8|2, 16| ![27e2a860-bbc9-11e8-b2bc-ac1f6b6435d0_8](https://i.imgur.com/HYiCC83.png)|Yes|No|No
c5f9de70-bba5-11e8-b2ba-ac1f6b6435d0_4|2, 16|![c5f9de70-bba5-11e8-b2ba-ac1f6b6435d0_4](https://i.imgur.com/IXvUMz4.png)|No|No|No

Notice that all images displayed above were labelled as '2|16' - Nucleoli and Cytosol, but only one of them (the first one) is correctly labelled. Applying the heuristics corresponding to both classes gave a really good indication of whether an image was correctly labelled as '2|16'. After repeating a similar process for a lot of images, I created a dataset of ~240,000 images and 144 unique labels. The dataset is public at [https://www.kaggle.com/novice03/clean-data](https://www.kaggle.com/novice03/clean-data).

For my highest scoring submission, I also trained a model on 26,000 slide-level images. These images were a balanced subset of the competition and public data. The above heuristics were not used to filter out slide-level images.

## Training

**First submission**

Architecture|Loss used|Dataset size|GPU used|Training time
|------------|----|------------|--------|-------------|
Efficientnet B3|BCE|26,000*|Tesla P100-PCIE-16GB|1hr
Inception v3|BCE|180,000**|Tesla P100-PCIE-16GB|1hr 40mins
Mobilenet v3|Focal Loss|180,000**|Tesla P100-PCIE-16GB|1hr 40mins

\* slide level dataset used to train Efficientnet B3

**Second submission**

Architecture|Loss used|Dataset size|GPU used|Training time
|------------|----|------------|--------|-------------|
Densenet 121|BCE|240,000|Tesla P100-PCIE-16GB|7hrs 30 mins
Inception v3|BCE|180,000**|Tesla P100-PCIE-16GB|1hr 40mins
Mobilenet v3|Focal Loss|180,000**|Tesla P100-PCIE-16GB|1hr 40mins

\** Inception and mobilenet were trained in google colab, which had much harder data storage constraints than Kaggle notebooks. Densenet was trained on kaggle. The 180,000 images were a subset of the 240,000 images.

The inception and mobilenet models used in both submissions are the same. For all models, resizing to 448x448, a random rotation of at most 270 degrees, and normalization was applied before training. During inference, 4x TTA was applied. For the first submission, the average of cell-level predictions and slide-level predictions was taken. For the second submission, the average of the all 3 cell-level models was taken. The weights of all 4 models can be found in the kaggle dataset in the ```weights``` folder. The inference notebook is available as ```inference.ipynb```.

## Conclusion 

To tackle the weak supervision problem, class-specific heuristics were applied on the cell-level dataset to extract a cleaner subset. For one submission 2 cell-level models were trained on this dataset. The predictions of these models were ensembled with a slide-level model. For the other submission, 3 cell-level models were ensembled. Interestingly, both submissions score very similarly. Despite the first submission having a slide-level model, it only scored 0.000036 more than the 3 cell-level models. This may show that **ensembling slide-level models with cell-level models improves performance very little when those cell-level models are trained on clean data with correct labels**. Furthermore, **when cell-level models are trained on clean data, they outperform other cell-level models trained on noisier and larger datasets (even when these models are ensembled with slide-level models).**
