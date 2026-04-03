# HONEY PRODUCTION FROM US BEES - ANALYSIS 
The scope of the project is to analyze the trend of honey produced by US honeybees along the period 1995-2021.
The code is presented using streamlit (execute "streamlit run run.py" for application).

## Run - 
git clone https://github.com/Ddav3/Bees_production_analysis.git
cd Bees_production_analysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run run.py

## Description
For a more detailed analysis, the project has been divided into 4 parts:
1. a first general analysis on the honey production
2. an estimate of the impact of pesticide use on the bees
3. an estimate of the effects of weather conditions on bees health
4. a short in-depth analysis on the jump recorded between 2009-2010.
The information are shown together with their related state. 
For a better transperency, the code is shown also in blocks of a jupyter notebook, which is also subdivided into the same parts and roughly explained. 


## Main Datasets
Here follows the links to the datasets that are mainly used for the analysis.  
- Honey Production dataset: https://www.kaggle.com/datasets/mohitpoudel/us-honey-production-19952021

- Apistox dataset: https://www.kaggle.com/datasets/baharabaz/apistox-dataset

- Weather's influence dataset: https://www.kaggle.com/datasets/jocelyndumlao/predicting-honeybee-health-from-hive-and-weather

- Varroa Destructor detection dataset: https://www.kaggle.com/datasets/anaisabelcaicedoc/varroa-detection-with-discrete-variables