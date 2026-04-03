import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_base_datasets,load_apistox_support_dataframe,load_varroa_detection_dataset, bar_single_ys_interactive, random_forest

#Code Part - 1 ---------------------------------------------------------------------------------------------
apistox_df = load_base_datasets(load_honey_production=False, load_weather_effects=False)[0]



#Varroa Destructor
varroa_detection_df = load_varroa_detection_dataset()

features = varroa_detection_df.drop(["WEIGHTED_SUM", "PREDICTED_ALERT"], axis = 1)
result_varroa = random_forest(
            features,
            varroa_detection_df["PREDICTED_ALERT"],
            n_estimators=640
            )[0]
varroa_detection_df["AVG_TEMPERATURE"] = varroa_detection_df["AVG_TEMPERATURE"].apply(lambda x: "20-25" if 20 <= x <= 25 else
                                                                "25-30" if 25 < x <= 30 else
                                                                "30-35" if 30 < x <= 35 else
                                                                "35-40" if 35 < x <= 40 else
                                                                "out_of_range")
varroa_detection_df["AVG_HUMIDITY"] = varroa_detection_df["AVG_HUMIDITY"].apply(lambda x: "35-50" if 35 <= x <= 50 else
                                                                "50-65" if 50 < x <= 65 else
                                                                "65-80" if 65 < x <= 80 else
                                                                "80+" if 80 < x else
                                                                "out_of_range")
#Application Part - 1 --------------------------------------------------------------------------------------
st.header("Jump Analysis and Colony Collapse Disorder")
st.subheader("Premise")
st.markdown("""
What has been presented till this point might, by itself, look like an amount of abstract observations. In reality, we can use properly what understood so far to provide and explanation to the abnormal jump in the production during the years 2009-2010.
\nThe truth is, the downfall is a direct consequence of a phenomenum called **Colony Collapse Disorder**: in short, due to various reasons, worker bees were forced to leave the hives, abandoning it together with the queen, the food and the brood, causing their ruin. This tragedy generated vaste consequences in the bees global population: considering the US, we are talking about *30-90%* (based on the zone) of the total amount of hives lost up to these days, with a loss of **60%** recorded in 2025 with respect to 2024. It is indeed a dangerous situation and is essentially due to numerous factors, mainly:\n
-   the propagation of a parasite, which is the :red-badge[Varroa Destructor], and a fungus, called :red-badge[Nosema Apis];
-   the exposure of bees to continued :red-badge[usage of pesticides];
-   the :red-badge[malnutrition] and presence of :red-badge[environmental stress factors], mainly due to climate change.
\nKnowing this, we can now try to create a picture of the situation of back when this crysis started (2006). We will see only few of the causes in detail.
            """)
st.image("assets/images/ccd.jpg", width="stretch")
st.subheader("Varroa Destructor Conditions Analysis")
st.markdown("""
Probably the main cause of the decline in the number of active hives, the :red-badge[Varroa Destructor] is a parasite that comes from Asia and that reached the US because of the imports of asian wood used for building the apiaries. In order to reproduce, females enters the brood cells, pierces the larva and lays from 1 to 6 eggs, which can cause, in a couple of weeks, to give birth to 3/4 Varroas, on average. Clearly, the bee larva dies in the process.\n
But the Varroa reached US a couple of decades before the CCD. In fact, the parasite was kept under control through some programs, which became less intensive right before 2006, causing the catastrphic propagation. Since then, beekepers have been careful on the matter, monitoring the hives. On this note, a study on the weather conditions as signs of alert for a Varroa Destructors' colony has been conducted. Here follow the results of a classification made with **Random Forest Classifier**, in order to establish which parameter represents a most significant warning sign:""")
st.plotly_chart(go.Figure(
    data=[go.Pie(
        labels=result_varroa.Features,
        values=result_varroa["Weight %"],
        hole=0.4,
        textinfo="percent",
    )]))

st.markdown("""If the temperature and the humidity are the main aspects to watch out for, then it is somehow interesting to understand under which conditions, for these two, the alert of a Varroa colony is higher. To do so, we can overwrite the specific values for both of them and categorize them into ranges, group them and sort the combination of ranges by the mean of the alert value. Here follows what just said:""")
st.dataframe(varroa_detection_df.groupby(["AVG_TEMPERATURE","AVG_HUMIDITY"])["PREDICTED_ALERT"].mean().sort_values(ascending=False))
st.markdown("""What we can conclude is simply that, **the lower the temperatures, but the higher the humidity, the higher is the alert**, which also explains why the stressors parameter happens to be higher for Utah compared to the North Carolina.""")

#Code Part - 2 ---------------------------------------------------------------------------------------------

#2005#
apistox_complete_2005_df = apistox_df.merge(load_apistox_support_dataframe(year = 2005, remove_kg=False))
apistox_complete_2005_df.drop(["name", "source", "SMILES", "CID","other_agrochemical", "year"], axis = 1, inplace=True)
apistox_complete_2005_df.columns = apistox_complete_2005_df.columns.map(str.upper)

quantity_part = apistox_complete_2005_df.groupby("STATE_NAME")[["MEAN_KG"]].sum().reset_index()
variety_part = apistox_complete_2005_df.groupby("STATE_NAME")[["INSECTICIDE","HERBICIDE", "FUNGICIDE"]].sum().reset_index()
usage_2005_df = quantity_part.merge(variety_part).set_index("STATE_NAME").sort_values("MEAN_KG", ascending=False)


#2019#
apistox_complete_2019_df = apistox_df.merge(load_apistox_support_dataframe(year = 2019, remove_kg=False))
apistox_complete_2019_df.drop(["name", "source", "SMILES", "CID","other_agrochemical", "year"], axis = 1, inplace=True)
apistox_complete_2019_df.columns = apistox_complete_2019_df.columns.map(str.upper)

quantity_part = apistox_complete_2019_df.groupby("STATE_NAME")[["MEAN_KG"]].sum().reset_index()
variety_part = apistox_complete_2019_df.groupby("STATE_NAME")[["INSECTICIDE","HERBICIDE", "FUNGICIDE"]].sum().reset_index()
usage_2019_df = quantity_part.merge(variety_part).set_index("STATE_NAME").sort_values("MEAN_KG", ascending=False)

usage_difference_CCD_df = usage_2019_df - usage_2005_df
#Application Part - 2 --------------------------------------------------------------------------------------
st.subheader("Pesticide Usage")
st.markdown("""
Other elements present a thread to the safety of bees colonies. It's also curious to denote that starting from 2006, various intensive campaigns have been launched, such as the **Pollinator Protection Plan (2007)**, meaning the states have been taking actions to face the crysis.
The prime examples were the states of California, Pennsylvania and Florida, but also some midwest states (Michigan, Wisconsin, Minnesota, North and South Dakota, etc) in later years, all with their initiatives.\n
On this note, we can take a look at the **changes** they made in the :red-badge[use of pesticide] since then (so up the 2019 dataset, as the most recent one), which is more or less a measure adopted by the majority. Here follows a summary of what just said, sorting the states from those who managed to reduce the usage the most:""")
st.dataframe(usage_difference_CCD_df.sort_values("MEAN_KG"))
st.markdown("""Note that the reduction for fungicide is lower than the rest, since their use is being somehow managed to face problems like the :red-badge[Nosema Fungus].""")

st.markdown("""The analysis stops here for now, but intentions for future updates are reported in the "**Future Works**" page.""")