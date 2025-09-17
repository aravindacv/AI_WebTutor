import streamlit as st
from utils import app_header, ensure_seed, progress_sidebar

st.set_page_config(page_title="Professor CoderBuddy AI — Tutor", layout="wide")

with st.sidebar:
    seed = st.number_input("Random seed (reproducible)", value=42, step=1)
ensure_seed(seed)

app_header("Professor CoderBuddy AI — Interactive Tutor",
           "Linear Algebra • Probability • Data Science\n\n"
           "Use the sidebar **Pages** to navigate units. Dark Mode is enabled 🌙.")

st.markdown("""
### How this works
- Each unit is its **own page** with sliders, inputs, and visuals.
- Use the **Mark this unit complete** button (sidebar) to track progress.
- All data is generated **in-code**; no external files needed.
""")

progress_sidebar()
st.markdown("---")
st.markdown("**Start with Unit 1 from the left sidebar → Pages. Enjoy learning!**")
