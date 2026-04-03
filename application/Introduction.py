import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

st.image("assets/images/bee.jpg", width="stretch")
st.header("Premise")
st.markdown("""
In recent years, there has been a continuous decline in global honey production (**19%** between 2020 and 2025 according to Food and Agricolture Organisation's data), even though its **market value** has **more than doubled**. This trend is primarily due to the living conditions of honey bees, that, for various reasons, are more and more compromised in terms of health and productivity.
\nHoney bees pollinate approximately **75%** of the world's food crops and are essential to the entire global ecosystem, since the production of fruits, vegetables, and seeds depends on their activity. However, their honey yields per hive have become increasingly scarce (from **65 lb/hive** in the 1990s to **45 lb currently**), and what we believe to be a minor inconvenience (having to buy honey at a high cost) is actually the result of a much larger problem.""")
st.subheader("Objective")
st.markdown("""
This project will analyze these dynamics through data of the **United States**, where the crisis has manifested particularly clearly. Specifically, the analysis will be carried out using python, plotting relevant information with plotly and doing trend predictions and feature classifications where necessary.
 """)