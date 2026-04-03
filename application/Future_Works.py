import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

#Code Part ---------------------------------------------------------------------------------------------

#Application Part --------------------------------------------------------------------------------------
st.header("Versions and Future works")
st.markdown("""
V1.0)
The projects analyze the incidence of :red-badge[pesticides] and :red-badge[weather] on the honey production downfall as indirect causes. However, future updates intend to add information related to the diffusion of the :red-badge[Nosema Fungus] and the involvement of :red-badge[pollution and climate change].
""")
