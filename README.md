# Signboard Detection 
Implementation of paper - [Automatic signboard detection and localization in densely populated developing cities](https://www.sciencedirect.com/science/article/abs/pii/S0923596522001369#preview-section-references)

## Introduction

Automatic signboard detection in a developing or less developed city is a difficult task due to its challenging urban features. In order to tackle challenges such as multiple signboard detection, heterogeneous background, variation in signboard shapes and sizes, we introduce a robust end to end trainable Faster R-CNN based model integrating new pretraining schemes and hyperparameter selection method. We demonstrate state-of-the-art performance of our proposed method on both [Street View Signboard Objects (SVSO)](https://zenodo.org/record/6865241) dataset and [Open Image Dataset](https://storage.googleapis.com/openimages/web/download.html). Our proposed method can detect signboards accurately (even if the images contain multiple signboards with diverse shapes and colors in a noisy background) achieving **0.90** mAP (mean average precision) score on **SVSO** independent test set.

**`This repository contains code for signboard detection which will return the segmented region of a signboard with localization details automatically.`**

## Graphical abstract
![Graphical abstract](https://user-images.githubusercontent.com/16709991/218386949-d81448df-deb3-4822-acd4-c4629fcf0c76.jpg)


## Installation

#### 1. Clone the repo.
```bash
git clone https://github.com/sadrultoaha/Signboard-Detection.git
cd Signboard-Detection
```

#### 2. Install the requirements.
Make sure that you have `Python <=3.7` installed on your system as the implementation uses `Tensorflow==1.15.2` and `Keras==2.2.4`. (see: [Tensorflow installation instructions](https://www.tensorflow.org/install))
```bash
pip install -r requirements.txt
```

## Signboard detector model weights and required files [**Required**]

Download model weights and required files from here - (**Download**: [Signboard_Detector.zip](https://drive.google.com/file/d/1Jcq6vEbAiJU9B0YVlQEMKhnIPdM-aThI/view?usp=share_link))

Unzip the 'Signboard_Detector.zip' file and place all the files and folders into the 'Signboard-Detection' directory.

## Usage

#### Run the signboard model on SVSO test data
```bash
python detection.py -test_file_path "Test" -output_csv output.csv -output_zip result.zip
```
#### Run the signboard model on Custom test data
To run the signboard model on your desired test data, set the test_file_path to the your own custom test dataset path.
```bash
python detection.py -test_file_path "your_test_data_path" -output_csv output.csv -output_zip result.zip
```

Arguments Details:
* -test_file_path: Path to the testing images folder, i.e., Path to the Public test data or Path to the Custom test data.
* -output_csv: Path to output the predicted localization and classification details.
* -output_zip: Path to output the segmented signboards on input images.

## Test Output
![image](https://user-images.githubusercontent.com/16709991/100400796-e1ac1e00-3081-11eb-9f15-7ab4d400514d.png)
![image](https://user-images.githubusercontent.com/16709991/100403087-13c07e80-3088-11eb-821f-ad88419293d8.png)


## Citation

```
@article{TOAHA2022116857,
title = {Automatic signboard detection and localization in densely populated developing cities},
journal = {Signal Processing: Image Communication},
volume = {109},
pages = {116857},
year = {2022},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2022.116857},
url = {https://www.sciencedirect.com/science/article/pii/S0923596522001369},
author = {Md. Sadrul Islam Toaha and Sakib Bin Asad and Chowdhury Rafeed Rahman and S.M. Shahriar Haque and Mahfuz Ara Proma and Md. Ahsan Habib Shuvo and Tashin Ahmed and Md. Amimul Basher},
keywords = {Object detection, Faster R-CNN, Clustering}
}

@dataset{toaha_md_sadrul_islam_2022_6865241,
  title        = {Street View Signboard Objects (SVSO)},
  month        = jul,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6865241},
  url          = {https://doi.org/10.5281/zenodo.6865241},
  author       = {Toaha, Md. Sadrul Islam and
                  Asad, Sakib Bin and
                  Rahman, Chowdhury Rafeed and
                  Haque, S. M. Shahriar and
                  Proma, Mahfuz Ara and
                  Shuvo, Md. Ahsan Habib and
                  Ahmed, Tashin and
                  Basher, Md. Amimul}
}

```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
* [https://github.com/yhenon/keras-frcnn](https://github.com/yhenon/keras-frcnn)
* [https://github.com/kbardool/Keras-frcnn](https://github.com/kbardool/Keras-frcnn)
    
</details>
