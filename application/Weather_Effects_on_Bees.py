import numpy as np
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_complete_inspections_on_weather_df, bar_single_ys_interactive, heatmap_cols_interactive

#Code Part ---------------------------------------------------------------------------------------------
inspections_on_weather_df = load_complete_inspections_on_weather_df()
inspections_on_weather_df_copy = inspections_on_weather_df.copy()

inspections_mean_result_df = inspections_on_weather_df.groupby("STATE")[[ "BROOD", "BEES", "QUEEN", "FOOD", "STRESSORS", "SPACE", "HEALTHY"]].mean()
mean_weather_influence_df = inspections_on_weather_df.groupby(["STATE", "YEAR"])[["PERCENT_MET", "TEMPERATURE", "HUMIDITY", "DEW_POINT", "WIND_SPEED", "WIND_GUST", "PRESSURE", "PRECIP"]].mean()

failed_rows_df = inspections_on_weather_df[inspections_on_weather_df.HEALTHY == 0].shape[0]
failed_params_df = inspections_on_weather_df[inspections_on_weather_df.HEALTHY == 0][["BROOD", "BEES", "QUEEN", "FOOD", "STRESSORS", "SPACE"]].sum()
failed_params_df = failed_rows_df - failed_params_df
bar_failures =  bar_single_ys_interactive(failed_params_df.index, failed_params_df.values, "Parameters", "Failures",
                    f"Params failure during inspections (over {inspections_on_weather_df[inspections_on_weather_df.HEALTHY == 0].shape[0]})")


inspections_on_weather_df = inspections_on_weather_df.drop(["STATIONID", "WEATHERID", "OBSID", "APIARYID", "HIVEID", "INSPECTIONID"],axis = 1)
inspections_on_weather_df = inspections_on_weather_df.groupby(["STATE", "YEAR", "MONTH"])

inspections_on_weather_df = inspections_on_weather_df.mean(numeric_only=True).round(3)
heatmap = heatmap_cols_interactive(inspections_on_weather_df, True)

#Application Part --------------------------------------------------------------------------------------
st.header("Weather Effects - Quick Overview")
st.image("assets/images/hive.jpg", width="stretch")
st.markdown("""
This third section is dedicated to the effects that the weather conditions can cause to the health conditions of the bees. The Dataset :green-badge[predicting-honeybee-health-from-hive-and-weather], of which you can learn more in the EDA section, is used to evaluate the possible influence of this factor. The inspections' result provided by the dataset indicates only the istant of the inspection, the weather recorded and the requirements met.
\nIn light of this, the first we can do is try to understand how often a failure is recorded because of certain parameters and which eventually are the parameters that fail the most. Here follows a representation of this:
""")
st.plotly_chart(bar_failures)
st.markdown("""Apparently, we could affirm the **_STRESSORS_** parameter is the one that mainly causes the inspections to fail. Unfortunately, this affirmation is kind of **weak**, having the dataset based on just two countries and with a small number of observations. Indeed, this outcome can be better understood if we examine the mean result from the inspections, for each state:
""")
st.dataframe(inspections_mean_result_df)
st.markdown("""
In fact, the bar plot shows a high rate of failure connected to the **_STRESSORS_** factor because of the low success rate during inspections in North Carolina.
\nHowever, even though such information may not help us in our research, there are few interesting affirmations we can confirm by looking at the data. First of all, it might be useful to visualize the possible relations between weather data and inspection parameters. Here follows a heatmap that represents this:
""")
st.plotly_chart(heatmap)
st.markdown("""
The fact the North Carolina shows a lower success rate for the inspections, in terms of weather effects, can be somehow explained if we think about the geography of the two states.\n
-   In Utah, there is a drier environment and higher ventilations, with stronger and fresher winds due to the slightly lower temperatures and the presence of more distributed mountain chains. Combined with the scarse precipitations, the state shows less humidity.
-   In North Carolina, the presence of the ocean coast causes the state to have a higher dew point, which causes major humidity and advantages parasites and fungi's propagation (see next section). 
""")
st.dataframe(mean_weather_influence_df)
col1, col2 = st.columns(2)
col1.image("assets/images/utah.jpg", width="content")
col2.image("assets/images/NC.jpg", )
st.markdown("""
\nWith all that considered, we now have a glimpse of what these data can tell us, even if an uncertain way. It's not to be excluded that with more data, stronger affirmations and conclusions could be made.
\nNevertheless, what we have now grants us the basis for what will be discussed in the section, where a :orange-badge[Jump Analysis] of the period :orange-badge[2009-2010] is carried out.
""")
