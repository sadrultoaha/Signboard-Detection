# Automatic Signboard Detection and Localization in Densely Populated Developing Cities
By Md. Sadrul Islam Toaha, Sakib Bin Asad, Chowdhury Rafeed Rahman, S. M. Shahriar Haque, Mahfuz Ara Proma, Md. Ahsan Habib Shuvo, Tashin Ahmed, Md. Amimul Basher

## Introduction
<!---**Automatic** Signboard Detection is based on [paper](https://arxiv.org/pdf/2003.01936.pdf), in this paper, we have solved the first step of auto establishment annotation problem by developing an automatic signboard detection system.-->
Automatic **signboard** detection in a developing or less developed city is a difficult task due to its challenging urban features. To approach this problem, we introduce a robust model with a new signboard [dataset](https://drive.google.com/drive/folders/1LQCgF3U-hPL46WLkq1dX8WJzRBSvCGga?usp=sharing). The signboard detection model is based on Faster R-CNN architecture including a smart proposal box generator algorithm and specialized pretrained techniques.

**This repository contains code for signboard detection which will return the segmented region of a signboard with localization details automatically.**
<!---## Architecture
![image](https://user-images.githubusercontent.com/16709991/100399693-eff83b00-307d-11eb-9a33-461ba104158c.png)-->

## Guideline for Google Colab Environment

### 1. Installation

#### Clone the repo.

Visit [Google Colab](https://colab.research.google.com), and clone the `Detection.ipynb` file from https://github.com/sadrultoaha/Signboard-Detection.git

#### Install the requirements.
```
%tensorflow_version 1.x
!pip install Keras==2.2.4
```
### 2. Usage.

#### Run the signboard model on SVSO test data
* Set the required path as follows and run all the notebook cells.
```
test_file_path = 'Test'
output_csv = 'output.csv'
output_zip = 'result.zip'
```
#### Run the signboard model on Custom test data
* Upload and connect your custom dataset into Google Colab environment.
* Set the required path as follows and run all the notebook cells.
```
test_file_path = 'your_test_data_path'
output_csv = 'output.csv'
output_zip = 'result.zip'
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
<!---
## Test Output
![image](https://user-images.githubusercontent.com/16709991/100400796-e1ac1e00-3081-11eb-9f15-7ab4d400514d.png)
![image](https://user-images.githubusercontent.com/16709991/100403087-13c07e80-3088-11eb-821f-ad88419293d8.png) -->
