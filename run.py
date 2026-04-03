import streamlit as st

st.title("HONEY PRODUCTION ANALYSIS")

pg = st.navigation([st.Page("application/Introduction.py"), 
                    st.Page("application/EDA.py"),
                    st.Page("application/Honey_Production.py"), 
                    st.Page("application/Apistox.py"), 
                    st.Page("application/Weather_Effects_on_Bees.py"),
                    st.Page("application/Jump_Analysis.py"),
                    st.Page("application/Future_Works.py")])
pg.run()