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
This is the dataset 3 that is modified so that it can be used with post-processing bias mitigation. It must be this dataset for post-processing bias mitigation to work. 
The content of the dataset and the labels are the same as the dataset 3. The labels were reassigned so that it works for the post-processing bias migitation. The instructions for replacing the dataset is available in. TODO: add link


### Fashion-MNIST Dataset. 

Fashion-MNIST is a drop in replacement dataset for the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This dataset consists of a training set of 60,000 examples and test set of 10,000 examples. Each example is an image of 28x28 grayscale and each example is associated with a label from one of the 10 cloasses. The 10 classes belongs to one of type of clothing. A detailed discribtion of the dataaset is available [here](https://github.com/zalandoresearch/fashion-mnist)

## Fairness tutorials

[Describe what one can do. What is available] 
Either copy paste the "read me" or link to it. 

### XAI tutorials
