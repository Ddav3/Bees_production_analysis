import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_base_datasets, load_complete_inspections_on_weather_df, load_varroa_detection_dataset, heatmap_cols_interactive

#Code Part ---------------------------------------------------------------------------------------------
honey_production_df, apistox_df = load_base_datasets(load_weather_effects=False)
weather_effect_df = load_complete_inspections_on_weather_df()
varroa_df = load_varroa_detection_dataset()
#Application Part --------------------------------------------------------------------------------------
st.header("Main Datasets EDA")
st.markdown("""This section is dedicated to the Exploratory Data Analysis (EDA) of the main Datasets involved in the project. What is shown corresponds to the version of the datasets after the minor, essential adjustments (for instance, before NaN removal or columns names fixing). Some relevant columns with no immediate meaning are briefly explained.""")
st.subheader("1. Honey Production Dataset")
st.markdown("""
The starting point of the analysis is the :yellow-badge[US_honey_production] dataset. It contains data for various states of the US with their production of honey from 1995 to 2021. There is also the value of production associated to the pounds of honey produced. 
\nSource: https://www.kaggle.com/datasets/mohitpoudel/us-honey-production-19952021""")
# st.markdown("**INFO**")
# #TODO
st.markdown("**DESCRIBE**")
st.dataframe(honey_production_df.describe())
st.markdown("**HEAD**")
st.dataframe(honey_production_df.head())
st.markdown("**POSSIBLE CORRELATIONS - numeric only**")
st.plotly_chart(heatmap_cols_interactive(honey_production_df.corr(numeric_only=True)))

st.subheader("2. Apistox Dataset")
st.markdown("""
The :blue-badge[apistox] Dataset has information about the toxicity level of various chemicals on honeybees. Among all columns, the **_CAS_** one is increadibly useful, since it contains the unique code of the chemical compound, known as **Chemical Abstracts Service Register Number (CASRN)**. Thanks to it, we have an alternative to the **_CID_** for recognizing compounds. In fact, it has been a reliable parameter during the realisation of the project, permitting to link :blue-badge[apistox] with :grey-badge[EPest_country_estimate]. 
Note that the voices **_INSECTICIDE_**, **_HERBICIDE_**, **_FUNGICIDE_** and **_OTHER_AGROCHEMICAL_** are booleans set to 1 wherever the compound is used for that product. 
\nSource: https://www.kaggle.com/datasets/baharabaz/apistox-dataset
""") 
st.markdown("**DESCRIBE**")
st.dataframe(apistox_df.describe())
st.markdown("**HEAD**")
st.dataframe(apistox_df.head())
st.markdown("**POSSIBLE CORRELATIONS - numeric only**")
st.plotly_chart(heatmap_cols_interactive(apistox_df.corr(numeric_only=True)))

st.subheader("3. Weather Effects on Bees Health Dataset")
st.markdown("""
This Dataset is composed of 6 tables that, combined, provide data for a possible correlation between the **weather conditions** and the **health of the bees**, which is evaluated through hourly HCC inspections with respect to certain parameters. Some tables show peculiar information.\n
:green-badge[**HCC_Inspections**]: the results of the inspections on the hives, evaluating a hive as HEALTHY whenever all requirements are set to 1 (the conditions are met).
-   **Brood**: the hives presents all stages of the brood (egg, larvae, pupae)
-   **Bees**: there are enough bees in the hive to manage it and defend it
-   **Queen**: there is a queen in the hive and she's young, healthy and can reproduce
-   **Food**: around the hive, there is enough source of food for the bees
-   **Stressors**: there are no particular stressors in the environment
-   **Space**: the hive is safe, clean, not in detriment and has enough space\n
:green-badge[**Hourly Weather**]: a hourly record of the weather conditions. Note that:
-   Temperature degrees are in Fahrenheit
-   **Wind_Gust** represents the peak of wind speed during the interval. 
\nSource: https://www.kaggle.com/datasets/jocelyndumlao/predicting-honeybee-health-from-hive-and-weather
""")
st.markdown("**DESCRIBE**")
st.dataframe(weather_effect_df.describe())
st.markdown("**HEAD**")
st.dataframe(weather_effect_df.head())
st.markdown("**POSSIBLE CORRELATIONS - numeric only**")
st.plotly_chart(heatmap_cols_interactive(weather_effect_df.corr(numeric_only=True)))


st.header("Other Datasets EDA") #---------------------------------------------------
st.markdown("""For further clarifications, here follows a subsection dedicated to other datasets involved.""")
st.subheader("a. Varroa Destructor Dataset")
st.markdown("""
The :red-badge[Varroa_Detection] dataset contains synthetic data used to identify the weather conditions under which there is a higher alert for a Varroa colony propagation. It's a base for training models with the specific scope of early detection. Among the parameters, there is a column, **_AVG_TVOC_**, which indicates the total amount of Volatile Organic Compounds in the air, on average.
\nSource: https://www.kaggle.com/datasets/anaisabelcaicedoc/varroa-detection-with-discrete-variables
""")
st.markdown("**DESCRIBE**")
st.dataframe(varroa_df.describe())
st.markdown("**HEAD**")
st.dataframe(varroa_df.head())
st.markdown("**POSSIBLE CORRELATIONS - numeric only**")
st.plotly_chart(heatmap_cols_interactive(varroa_df.corr(numeric_only=True)))
