import numpy as np
import pandas as pd 
import seaborn as sns
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_base_datasets, plot_all, linear_regression_and_plot, plot_multiple

#Code Part ---------------------------------------------------------------------------------------------
honey_production_df = load_base_datasets(load_apistox=False, load_weather_effects=False)[0]

for state in honey_production_df.state.unique():
    state_production = honey_production_df.loc[honey_production_df.state == state]
    #keeping states that have data for all the 27 years from 1995 to 2021
    if len(state_production) < 27:
        honey_production_df.drop(state_production.index, inplace=True)
plots_productions = plot_all(honey_production_df,"state", "year", "yield_per_colony")

years = honey_production_df['year'].unique()

production_means = honey_production_df.groupby("year")["production"].mean()
production_prediction = linear_regression_and_plot(x_array=years, 
                                                    pred_x_array=np.concatenate([years,np.arange(2022, 2032)]), 
                                                    interactive= True,
                                                    y_array=production_means,
                                                    train_size=0.8, 
                                                    random_state=30, 
                                                    ylabel = "Mean Annual Honey Production")

production_stock_plot = plot_multiple(honey_production_df.year.unique(),
                                      honey_production_df.groupby("year")[["stocks", "production"]].mean(),
                                      xlabel= "Years")
value_production_plot = plot_multiple(honey_production_df.year.unique(),
                                      honey_production_df.groupby("year")["value_of_production"].mean(),
                                      xlabel="Years")

value_production_means = honey_production_df.groupby("year")["value_of_production"].mean()
value_production_prediction = linear_regression_and_plot(x_array = years, 
                                                         pred_x_array=np.concatenate([years,np.arange(2022, 2032)]),
                                                         interactive=True,
                                                         y_array=value_production_means,
                                                         train_size=0.8,
                                                         random_state=30,
                                                         xlabel="Year",
                                                         ylabel = "Mean Annual Honey Price")
#Application Part --------------------------------------------------------------------------------------

st.header("Analysis")
st.markdown("""
First, what we want to do is to see how much the honey production has changed over the years in US. To this end, the dataset :yellow-badge[US_honey_production.csv] is used in order to display, over the period 1995-2021, this trend.
For a more clear view, here follows a global representation for all the states of which we have data from 1995 and 2021:
""")
st.pyplot(plots_productions)

st.subheader("Production Trend Prediction")
st.markdown(
"""
Now we need a way to represent all this information. The most basic idea is indeed to take the amount of honey produced by all states in the same year, compute the mean (repeated for each year) and show the results in a scatter plot. However, since we have this information, it would be nice to take advantage of it and try to predict the trend up to our days.\n
Here follows a graphical representation of what was just said, using the linear regression method for prediction (notice that the data chosen for training/testing are highlighted differently):  
"""
)
st.plotly_chart(production_prediction, width="stretch")
st.markdown(
"""
Clearly, if we project our view to 2026 (the year during which the code is being written), we cannot affirm that this is a valid estimate because of the jump between 2009 and 2010.
However, instead of discarding this data, we can examine the production along with data regarding the value of the production (price).
"""
)
st.subheader("Stock-Price Analysis and Price Trend Prediction")
st.markdown(
"""
As just mentioned, it will be interesting to see the evolution in the quantity of honey produced and in the actual price at which it's evaluated. Here follows a representation of this, which not only considers production and value of production (cost), but also illustrates the difference between **honey produced** and **stocks kept** for sales:
"""
)
st.plotly_chart(production_stock_plot, width="stretch")
st.markdown(
"""Curiously enough, the trend is divided in two parts:\n
-   before 2009: the production far surpasses the sole amount of honey that is kept by companies for profit, while the rest is kept and sold directly by beekeepers. In any case, the quantities are more or less proportional to each other;
-   from 2010: the 2 quantities coincide, meaning the quantity of honey produced by hives is all given to enterprises and sol in the market.\n
To what happened in the gap 2009-2010 is dedicated a specific analysis has been conducted (see Section "Gap Analysis").
\nWhat we do now is consider these information and, in light of them, the trend of the global cost (defined as value of production) of honey. This value. This value is computed has the mean price per pound times the produced quantity. Note that the dataset does not speak about the influence of inflation, yet we expect the value to get higher and higher. Here follows what just explained, showing, then, the total mean price per year:
"""
)
st.plotly_chart(value_production_plot, width = "stretch")
st.markdown(
"""
The trend is slightly more linear than the other, yet there are few evident jumps. Nevertheless, we can still try to make a prediction with the data we have. Here follows a representation of the estimate for the future years up to 2031 (5 years from now) 
"""
)
st.plotly_chart(value_production_prediction, width ="stretch")
st.markdown(
"""
We don't know the actual quantity of honey that has been produced in these days, we are just aware of the fact that a pound of honey costs from 8 and 15 dollars, considering only honey produced inside the US territory (not imported). The chart shows an estimate of approximately 10.290k dollars, meaning the assumed quantity of honey produced shal vary between 1.286k and 686k (just divide the total for 2026 with the cost for a pound).
\nThat concludes the general overview. In the next sections, it will be shown:\n
-   the toxicity level from :blue-badge[apistox.csv] combined with other information;
-   a glimpse of the effect caused by the weather to the bees health, combining the tables of the dataset :green-badge[predicting-honeybee-health-from-hive-and-weather].  
"""
)