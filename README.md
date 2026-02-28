# BEES PRODUCTION ANALYSIS 
The code analyses a dataset containing info mainly related to the production of honey from bees' colonies situated in US countries from 1995 to 2021. 
The dataset is examined showing the evolution in the production of honey and the number of colonies over the years and depending on the area. After that, a prediction on the possible number of colonies/quantity of honey produced for the next 10 years is realized through a linear regression model. 

This main dataset is then combined with two other datasets.

The first one containes information regarding the toxicity of pesticides and agrochemicals to honeybees, in order to predict whether the compound is toxic or not for them. 
By merging these two, the code checks the correlation between the pesticides adopted in countries in which the use is the highest and the possible reduction of colonies in these countries.

The second one shows the weather's effects on the beehives in some countries of US. The analysis is made based on hourly observations on apiaries during 2016-2017.
The combination of the two datasets aims at understanding the influence of weather and temperatures on the change in beehive number, other than being able to classify a beehive as "healthy"/"unhealthy", based on the given data.

The content is presented using streamlit. 

## Datasets
- honey production dataset: https://www.kaggle.com/datasets/mohitpoudel/us-honey-production-19952021

- apistox dataset: https://www.kaggle.com/datasets/baharabaz/apistox-dataset

- weather's influence dataset: https://www.kaggle.com/datasets/jocelyndumlao/predicting-honeybee-health-from-hive-and-weather