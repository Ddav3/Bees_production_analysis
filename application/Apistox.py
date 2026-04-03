import numpy as np
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_base_datasets,load_apistox_support_dataframe, random_forest

#Code Part ---------------------------------------------------------------------------------------------
apistox_df = load_base_datasets(load_honey_production=False, load_weather_effects=False)[0]
pesticide_usage_df = load_apistox_support_dataframe(remove_kg=False)

apistox_complete_df = apistox_df.merge(pesticide_usage_df)
apistox_complete_df.drop(["name", "source", "SMILES", "CID","other_agrochemical", "year"], axis = 1, inplace=True)
apistox_complete_df.columns = apistox_complete_df.columns.map(str.upper)
apistox_complete_df_copy = apistox_complete_df.copy()

label_results,_  = random_forest(apistox_complete_df[["HERBICIDE", "FUNGICIDE", "INSECTICIDE", "TOXICITY_TYPE"]],
                    apistox_complete_df.LABEL,
                    n_estimators=750,
                    random_state=42)

ppdb_level_results,_  = random_forest(apistox_complete_df[["HERBICIDE", "FUNGICIDE", "INSECTICIDE", "TOXICITY_TYPE"]],
                    apistox_complete_df.PPDB_LEVEL,
                    n_estimators=750,
                    random_state=42)

# Variety 
## toxicity label 
apistox_complete_df = apistox_complete_df.groupby("STATE_NAME")[["INSECTICIDE","HERBICIDE", "FUNGICIDE"]].sum()
apistox_complete_df["ORDER"] = 0
label_results_feature_indexed = label_results.set_index("Features")
for index in range(3):
    apistox_complete_df["ORDER"] += apistox_complete_df.iloc[:,index] * float(label_results_feature_indexed.iloc[index, 0])
apistox_complete_df = apistox_complete_df.sort_values("ORDER", ascending=False).drop("ORDER", axis = 1)
variety_order_label = apistox_complete_df.reset_index()["STATE_NAME"].rename("VARIETY_ORDER")

## ppdb level 
apistox_complete_df = apistox_complete_df_copy

apistox_complete_df = apistox_complete_df.groupby("STATE_NAME")[["INSECTICIDE","HERBICIDE", "FUNGICIDE"]].sum()
apistox_complete_df["ORDER"] = 0
ppdb_results_feature_indexed = ppdb_level_results.set_index("Features")
for index in range(3):
    apistox_complete_df["ORDER"] += apistox_complete_df.iloc[:,index] * float(ppdb_results_feature_indexed.iloc[index, 0])

apistox_complete_df = apistox_complete_df.sort_values("ORDER", ascending=False).drop("ORDER", axis = 1)
variety_order_ppdb = apistox_complete_df.reset_index()["STATE_NAME"].rename("VARIETY_ORDER")


# Quantity
apistox_complete_df = apistox_complete_df_copy
label_q_results, _ = random_forest(apistox_complete_df[["HERBICIDE", "FUNGICIDE", "INSECTICIDE", "TOXICITY_TYPE", "MEAN_KG"]],
                    apistox_complete_df.LABEL,
                    n_estimators=750,
                    random_state=42)
quantity_order_label = apistox_complete_df.groupby("STATE_NAME")[["MEAN_KG"]].sum().sort_values("MEAN_KG", ascending=False).reset_index()["STATE_NAME"].rename("QUANTITY_ORDER")

ppdb_q_results, _ = random_forest(apistox_complete_df[["HERBICIDE", "FUNGICIDE", "INSECTICIDE", "TOXICITY_TYPE", "MEAN_KG"]],
                    apistox_complete_df.PPDB_LEVEL,
                    n_estimators=750,
                    random_state=42)
quantity_order_ppdb = apistox_complete_df.groupby("STATE_NAME")[["MEAN_KG"]].sum().sort_values("MEAN_KG", ascending=False).reset_index()["STATE_NAME"].rename("QUANTITY_ORDER")


#Comparison
#first, for toxicity label
usage_ranking_df = pd.concat([variety_order_label, quantity_order_label], axis = 1)
variety_order_idx = usage_ranking_df.index.to_list()
quantity_order_idx = []
for state in usage_ranking_df.VARIETY_ORDER:
    quantity_order_idx.append(usage_ranking_df[usage_ranking_df["QUANTITY_ORDER"] == state].index[0])

usage_ranking_df["POS_DIFF"] = np.abs(pd.Series(variety_order_idx) - pd.Series(quantity_order_idx))

conditions = [
    usage_ranking_df["POS_DIFF"] == 0,
    usage_ranking_df["POS_DIFF"] <= 3, 
    usage_ranking_df["POS_DIFF"] <= 5,
    usage_ranking_df["POS_DIFF"] <= 10,
]
distance_labels = ["SAME POSITION", "VERY CLOSE", "CLOSE", "DISTANT"]


usage_ranking_df["POS_DIFFERENCE"] = np.select(conditions, distance_labels, "VERY DISTANT")
usage_ranking_label_df = usage_ranking_df.drop("POS_DIFF",axis=1)

#second, for ppdb level
usage_ranking_df = pd.concat([variety_order_ppdb, quantity_order_ppdb], axis = 1)
variety_order_idx = usage_ranking_df.index.to_list()
quantity_order_idx = []
for state in usage_ranking_df.VARIETY_ORDER:
    quantity_order_idx.append(usage_ranking_df[usage_ranking_df["QUANTITY_ORDER"] == state].index[0])

usage_ranking_df["POS_DIFF"] = np.abs(pd.Series(variety_order_idx) - pd.Series(quantity_order_idx))

conditions = [
    usage_ranking_df["POS_DIFF"] == 0,
    usage_ranking_df["POS_DIFF"] <= 3, 
    usage_ranking_df["POS_DIFF"] <= 5,
    usage_ranking_df["POS_DIFF"] <= 10,
]
distance_labels = ["SAME POSITION", "VERY CLOSE", "CLOSE", "DISTANT"]

usage_ranking_df["POS_DIFFERENCE"] = np.select(conditions, distance_labels, "VERY DISTANT")
usage_ranking_ppdb_df = usage_ranking_df.drop("POS_DIFF",axis=1)

#Application Part --------------------------------------------------------------------------------------
st.header("Apistox - Analysis and Classification of Pesticides")
st.markdown("""
Now we take a look at the second part, which involves the analysis on the potential harm of chemical compounds recorded in the :blue-badge[apistox.csv] dataset. The dataset alone is actually unusable for our scope: for this reason, information about the :grey-badge[EPest_country_estimate] directly provided by the US government is used to provide the following data. However, the two datasets alone were not sufficient to make a link (due to inconsistencies in the use of names). That's why a further dataset was necessary: :violet-badge[Chemical List PPDB], which is updated to the data of download.\n
Having said that, the first step is to correctly understand the result provided by the :blue-badge[apistox.csv] dataset, both in terms of\n
-   **determination of toxicity**, meaning whether the compound is harmful for bees(1) or not(0);
-   **level of toxicity**, from **not necessarily harmful** (0) to **heavily harmful** (2).
The main features involved are: INSECTICIDE, HERBICIDE, FUNGICIDE and TOXICITY TYPE.
For a minimal understanding of the features involved, see the "**EDA**" section, where the apistox dataset is explained.
Given the type of analysis, two slightly different approaches are followed.
""")
st.subheader("Variety of Pesticides approach")
st.markdown("""
The first chosen approach is based on the premise that *"The more types of pesticides the bees are exposed to, the larger the impact and the damage dealt to the colonies"*. With this concept, let's start by seeing which are the features that mainly influence the **determination of toxicity**. Here follows the result of the classification made using **Random Forest Classifier**:
""")
st.dataframe(label_results)
st.markdown("""
Now we do the same, but changing the target, in order to see what determines the **level of toxicity** the most:
""")
st.dataframe(ppdb_level_results)
st.markdown("""
In the end, it's no surprise this conclusion is reached: if the compound is mainly used in insecticide, it's strongly possible it will cause harm to bees.
In the notebook version of this project, there's also a complex of bar plots that sorts the states from the ones with the higher usage of different compounds to those with a minor amount, over the years. 
""")

st.subheader("Quantity of Pesticides approach")
st.markdown("""
The second approach follows the idea that *"the more pesticide you use, in quantity, the heavier the consequences for the bees health"*. This approach is computed in a very similar way, but the column MEAN_KG is added among the features. Keeping the same order of things, here follows the classification of features in the **determination of toxicity**: 
""")
st.dataframe(label_q_results)
st.markdown("""
It's to be expected that a compound used for insecticides would be the main reason of harm for bees. However, the new information regarding the quantity seems to be more influent than the fact we are using other types of pesticide. 
\nNow follows the classification of features in the **establishment of the level of toxicity**:
""")
st.dataframe(ppdb_q_results)
st.markdown("""
These results do make sense, since we also expect a compound to be more and more harmful, as the quantity deployed increases, unlike the previous classification, where we only wanted to know whether a compound was dangerous or not. Then again, it's enough to affirm that huge amount of pesticides (no matter their scope) only bring heavy harm ot the colonies. 
""")
st.subheader("Comparison between approaches")
st.markdown("""
In conclusion of this part, we see how much differs the listing of the states considering the two approaches (from the ones with higher variety/quantity of pesticide used to those with lower ones). Also, another column is added: it defines, considering the position of state in the variety order, the distance in the ordering by quantity. Here follows what just explained, showing, respectively, the ranking for **toxicity label** and **ppdb level**: 
""")
st.dataframe(usage_ranking_label_df)
st.dataframe(usage_ranking_ppdb_df)
st.markdown("""
The two results are almost identical, aside from a couple of cases (rows 5 and 7, rows 27 and 28).
At any rate, there are indeed cases where a state shows a relevant distance from a column to the other, as well as cases in which the position stays the same, showing that the approach we consider is important in the ranking.
\nWith this, we conclude the second part of the analysis and prepare to proceed with the third part, about :green-badge[predicting-honeybee-health-from-hive-and-weather].
""")

# mask = (usage_ranking_ppdb_df[1:2].astype(str) != usage_ranking_label_df[1:2].astype(str))
# styled_df1 = usage_ranking_ppdb_df.style.apply(
#     lambda s: ['background-color: lightcoral' if mask[s.name].iloc[i] else '' for i in range(len(s))],
#     axis=1
# )
# st.dataframe(styled_df1)