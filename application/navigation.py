import streamlit as st

pg = st.navigation([st.Page("introduction_page.py"), 
                    st.Page("honey_production_page.py"), 
                    st.Page("apistox_page.py"), 
                    st.Page("weather_effect_on_bees_page.py"),
                    st.Page("gap_analysis.py")])
pg.run()