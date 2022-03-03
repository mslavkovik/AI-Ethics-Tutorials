# AI-Ethics-Tutorials

This is a collection of tutorials on explainability and fairness methods. They are curated and simplified by Than Htut Soe and Marius Pedersen. 


## Datasets in the tutorial
The tutorial use X datasets. 

### DATASET1- Adult dataset decription and link
This data was extracted from the census bureau database (USA) but can now be accessed through the UCI archive:
https://archive.ics.uci.edu/ml/datasets/adult 
from 1996
The dataset classifies whether people make more or less than 50K$ a year based on features they have availble at the census bureau.

For a more detailed description: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/Adult_dataset/adult%20description.txt 
For a tutorial: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/Adult_dataset/Adult%20dataset.ipynb

For a Lime tutorial using this dataset: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/Lime%20tutorial/Lime%20explanations%20on%20adult%20dataset.ipynb

### DATASET2- Portugal students
This data was extracted by Paulo Cortez, University of Minho, GuimarÃ£es, Portugal, http://www3.dsi.uminho.pt/pcortez
And can be accesed here: https://archive.ics.uci.edu/ml/datasets/student+performance 

This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details).

For a detailed description: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/Portugal_students/portugal-students.txt
For a tutorial: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/Portugal_students/Portugal.ipynb

### DATASET3 - SOUTH GERMAN CREDIT Data
This dataset classifies people described by a set of attributes as good or bad credit risks. The data originates from Häußler (1979, 1981) and Fahrmeir and Hamerle (1981, 1984) and is sampled from 	1973-1975 and comes from a large regional bank in Southern Germany. This is a updated version
	from Grömping, U. (2019) and more can be read on it from the statistics and the background here:
	http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf 
  
For a detailed description: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/SouthGerman_tutorial/South_german_description.txt
For a tutorial: https://github.com/mslavkovik/AI-Ethics-Tutorials/blob/main/Tutorials/SouthGerman_tutorial/SGerman_credit.ipynb

DATASET N - decroption and link 

### DATASET4 - South German Credit Data - Modified
This is the dataset 3 that is modified so that it can be used with post-processing bias mitigation. The original South German Credit dataset must be replaced with this version for Fair 2 and Fair 3 tutorials to work. 
To download this modified version [click here](Tutorials/Fair_tutorials/german). The instructions on how to replace the original South German Dataset is available in the [Fair 2 notebook.](Tutorials/Fair_tutorials/0_in_profess_fair.ipynb)

The content of the dataset and the labels are the same as the dataset 3. The labels were reassigned so that it works with the aif360 toolkit. 


### Fashion-MNIST Dataset. 

Fashion-MNIST is a drop in replacement dataset for the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This dataset consists of a training set of 60,000 examples and test set of 10,000 examples. Each example is an image of 28x28 grayscale and each example is associated with a label from one of the 10 cloasses. The 10 classes belongs to one of type of clothing. A detailed description of the dataaset is available [here](https://github.com/zalandoresearch/fashion-mnist)

## Fairness tutorials

[Describe what one can do. What is available] 
Either copy paste the "read me" or link to it. 


All the fairness tutorials (that Than's has made) are built on the German Dataset 

There are three tuorials for fairness as follows:
1. German Credit Dataset analysis using Aequitas [link](Tutorials/Fair_tutorials/0_Aequitas%20German%20Dataset.ipynb)
2. Performing in processing bias migitagation with Gerry Fair Classifier on German Credit Dataset  [link](Tutorials/Fair_tutorials/0_in_profess_fair.ipynb)
3. Performing post processing bias migitagation with calibrated odds-equalizing post-processing algorithm on the German Credit Dataset [link](Tutorials/Fair_tutorials/0_postprocessing_fair.ipynb)

Each of the fairness tutorials are described in deails in their own paragraphs as follows.
All toutirals should be self documented in their own Juypter notebooks. 
Note: To run tutorial 2 and 3  [common_utils.py](Tutorials/Fair_tutorials/common_utils.py) must be in the same folder.

### Fair 1 - German Credit Dataset analysis using Aequitas
Dataset: German Dataset
Required packages: aequitas, aif360
AIF360 is used for loading the German Dataset easily with the API. 
Installing prerequisite: pip install aequitas aif360, conda install -c conda-forge aequitas aif360

This tutorial explores the German Dataset with some plotting, train a simple SVC model and then audit the results of the SVC model using Aequitas. First, the bias in the dataset for age and sex is visually represented by plotting the credit approval rates between older and younger people as well as male and female. Visually, it can be seen that a higher portion of older and male datapoints are approved for credit. 

[Aequitas](https://github.com/dssg/aequitas) is built to audit the output of machine learning models. Therefore, the column with name "score" is created with the score from the machine learning model we just trained. To find out the biases in the machine learning data ii fmachine learning model, the origial dataset with the "score" column is necessary. Therefore, auditing fairness for the disparity tolerence between age groups and sex groups are performed. 

[Access the tutorial here](Tutorials/Fair_tutorials/0_Aequitas%20German%20Dataset.ipynb)

### Fair 2 - Performing in-processing bias migitagation with Gerry Fair Classifier on German Credit Dataset
Dataset: German Dataset (modified German Dataset is required. Check the notebook for instructions)
Required packages: aif360, fairlearn, sklearn
Installing prerequisite:  conda install -c conda-forge aif360 fairlearn scikit-learn 

This tutorial uses Gerry Fair Classifier to do in-processing bias migitation on German dataset. First, install the necessary packages. After that, copy our version of German Credit Dataset into the specified folder to replace the existing one. In this example, we will use the Gerry Fair Classifier to do in-processing bias mitigation for age. First, the bias in the age is demonstrated by calculating the statistical parity differences between privileged group (older people) and unprovileged group younger people. Then we will demonstrate the consequences of not using any bias migitation measure by training the SVM classifier and calculating statistical parity differences. 

To demonstrate the usage of GerryFairClassifier to perform bias migitation, the same method, SVM classifier, is trained but this time with a GerryFairClassifier. Then the statistical differences between unprivileged and privileged group is computed again. The bias demonstration is shown by reduction in stastical parity difference. 

[Access the tutorial here](Tutorials/Fair_tutorials/0_in_profess_fair.ipynb) Note: To run this tutorial [common_utils.py](Tutorials/Fair_tutorials/common_utils.py) must be in the same folder as this notebook.

### Fair 3 - Performing post-processing bias migitagation with calibrated odds-equalizing post-processing algorithm on the German Credit Dataset
Dataset: German Dataset (modified German Dataset. Check the notebook from Gerry Fair Classifier to use the modified dataset)
Requierd packages: aif360, fairlearn, sklearn
Installing prerequisites:  conda install -c conda-forge aif360 fairlearn scikit-learn

This this toturial uses post-processing bias mitigation to remove age bias in an SVM classifier trained with the German Credit Dataset. The first two steps of this tutorial are the same as the previous tutorial. That is demonstrating thebias in the dataset and how using this dataset without any bias mitigation can lead to biased classifier. However, we use a different machine learning method in this one which is LogisticRegression. The dataset had to be split into training, testing and validation set validation set is required for the mitigation method. Three fairness measures are used in this example. These are GFPR, GFNR and statistical parity difference (mean difference). 

CalibratedEqOddsPostprocessing method is then used to apply post-processing bias mitigation and the three measures are used to demonstrate it. It is to be noted however that the CalibratedEqOddsPostprocessing method do have a cost_constraint which can be fnr (False Negative Rate), fpr (False Positive Rate) and weighted which is the mean of fnr and fpr. 

[Access the tutorial here](Tutorials/Fair_tutorials/0_postprocessing_fair.ipynb) Note: To run this tutorial [common_utils.py](Tutorials/Fair_tutorials/common_utils.py) must be in the same folder.


### XAI tutorials

## Explainability Tutorials with SHAP
There are two explainability tutorials. They are as follows:
1. Census income classification with LightGBM [link](Tutorials/Explain_tutorials/SHAP_Fashion_Mnist_Explainer.ipynb)
2. Fashion-MNIST SHAP DeepExplainer  [link](Tutorials/Explain_tutorials/SHAP_Census%20income%20classification%20with%20LightGBM.ipynb)

Each tutorial is descirbed in their own sections in details

### XAI 1 - Census income classification with LightGBM
Dataset: Adult Income dataset 
Required packages: shap, lightgbm
Installing prequisities: conda install -c conda-forge shap lightgbm

LightGBM is a tree-based ensemble method. In this tutorial, TreeExplainer of SHAP library to explain the decisions of the trained LightGBM model. The tutorial has two main steps: training the LightGBM model, and applying the TreeExplainer of the SHAP library. Adult Income Dataset is used to train a LightGBM predictor to predicts where a sample earns more than 50,000 USD in a year. ThreeExplainer is then used to compute the SHAP values of the trained predictor. The SHAP values can then be used to explain visualizations of the predictor. It is done through visualizations of the SHAP values. The visualizations are: force plot for a single prediction, force plot for first 1000 predictions, and summary plots. Addition dependencies plots are also avaiable in the notebook. 

[Access the tutorial here](Tutorials/Explain_tutorials/SHAP_Fashion_Mnist_Explainer.ipynb)

### XAI 2 - Fashion-MNIST SHAP DeepExplainer
Dataset: Fashion-MNIST 
Required packaeges: shap, tensorflow
Installing prequisites: pip install tensorflow==2.5.0 shap
Currently the tensor flow version has to be 2.5.0 as the DeepExplainer will not work if tensorflow version is the latest one as required methods were removed after tensorflow 2.5.0.

In this tutorial, tensorflow with Kreas is used to create a simple convolutional neural network and trained that network to recognize 10 classes of clothings. Though it is a relative simple model, the training can take hours on a computer without hardware accelerated training. Therefore it is recommended just to load the trained model that is available in " fashion_mnist.zip". After the model is trained or loaded from the files, DeepExplainer of SHAP library is used to create SHAP values. The SHAP values are then visualized using image_plot on the subset of the Fashion-MNIST dataset. 

[Access the tutorial here](Tutorials/Explain_tutorials/SHAP_Census%20income%20classification%20with%20LightGBM.ipynb) 
