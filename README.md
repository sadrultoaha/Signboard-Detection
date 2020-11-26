# Automatic Signboard Detection using Segmentation Algorithm and Appropriate RoI Proposals
By Md. Sadrul Islam Toaha, Sakib Bin Asad, Chowdhury Rafeed Rahman, S. M. Shahriar Haque, Mahfuz Ara Proma, Md. Ahsan Habib Shuvo, TashinAhmed, Md. Amimul Basher

## Introduction
**Automatic** Signboard Detection is based on [paper](https://arxiv.org/pdf/2003.01936.pdf), in this paper, we have solved the first step of auto establishment annotation
problem by developing an automatic signboard detection system. While signboard detection in a developing country is a difficult task due to its challenging urban features. To approach this problem, we introduce a robust model with a new signboard [dataset](https://drive.google.com/drive/folders/1LQCgF3U-hPL46WLkq1dX8WJzRBSvCGga?usp=sharing). The signboard detection model is based on Faster R-CNN architecture including a smart proposal box generator algorithm and specialized pretrained technique.

This repository contains code for signboard detection which will return the segmented region of a signboard with localization details automatically.
## Architecture
![image](https://user-images.githubusercontent.com/16709991/100394977-4c049480-3069-11eb-8414-2e8422709086.png)

## Guideline for Google Colab Environment

### 1. Installation

#### Clone the repo.
Visit [Google Colab](https://colab.research.google.com)), and 
```
git clone https://github.com/sadrultoaha/Signboard-Detection.git
cd Signboard-Detection
```
#### Install the requirements.
make sure that you have `Tensorflow==1.15.2` and `Keras==2.2.4` installed. (see: [Tensorflow installation instructions](https://www.tensorflow.org/install))
```
!pip install -r requirements.txt
```
### 2. Usage.

#### SVSO test data
To run the signboard model on SVSO test data, download the test dataset from here : [Download](https://drive.google.com/drive/folders/1G47_PsJJWd0CleXynYqZuLWDPoNu2CCJ?usp=sharing).
#### Custom test data
To run the signboard model on your desired test data, create a folder and store your own custom test dataset inside the folder.
#### Run the signboard model
```bash
python detect.py -test_file_path Signboard-Detection/Test Data -output_csv output.csv -output_zip result.zip
```

## Guideline for Local Environment

### 1. Installation

#### Clone the repo.
```bash
git clone https://github.com/sadrultoaha/Signboard-Detection.git
cd Signboard-Detection
```
#### Install the requirements.
make sure that you have `Tensorflow==1.15.2` and `Keras==2.2.4` installed. (see: [Tensorflow installation instructions](https://www.tensorflow.org/install))
```bash
pip install -r requirements.txt
```
### 2. Usage.

#### SVSO test data
To run the signboard model on SVSO test data, download the test dataset from here : [Download](https://drive.google.com/drive/folders/1G47_PsJJWd0CleXynYqZuLWDPoNu2CCJ?usp=sharing).
#### Custom test data
To run the signboard model on your desired test data, create a folder and store your own custom test dataset inside the folder.
#### Run the signboard model
```bash
python detect.py -test_file_path "Test Data" -output_csv output.csv -output_zip result.zip
```
Arguments Instruction:
* -test_file_path: Path to the testing images folder, i.e., Path to the Public test data or Path to the Custom test data.
* -output_csv: Path to output the predicted localization and classification details.
* -output_zip: Path to output the segmented signboards on input images.

## Test Output
![image](https://user-images.githubusercontent.com/16709991/100394939-1fe91380-3069-11eb-845a-31f55fbbe99e.png)



