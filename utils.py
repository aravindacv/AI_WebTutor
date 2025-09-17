import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app_header(title, subtitle=None):
    st.markdown(f"# 👩‍🏫 {title}")
    if subtitle:
        st.markdown(subtitle)

def real_world_box(text):
    st.info(f"**Real-World Link:** {text}")

def try_this_box(text):
    st.success(f"**Try This!:** {text}")

def quiz_block(qs, ans):
    with st.expander("✅ Mini Quiz — Click to reveal answers"):
        for i, (q, a) in enumerate(zip(qs, ans), start=1):
            st.write(f"**Q{i}.** {q}")
            st.markdown(f"- **Answer:** {a}")

def ensure_seed(seed):
    if seed is not None:
        try:
            np.random.seed(int(seed))
        except Exception:
            pass

def init_progress():
    if "unit_done" not in st.session_state:
        st.session_state.unit_done = {i: False for i in range(1,6)}

def progress_sidebar(active_unit=None):
    init_progress()
    with st.sidebar:
        st.title("Progress Tracker")
        done = sum(st.session_state.unit_done.values())
        st.progress(done/5.0, text=f"{done}/5 units complete")
        for i in range(1,6):
            label = f"Unit {i}"
            checked = st.session_state.unit_done[i]
            st.checkbox(label, value=checked, key=f"chk_{i}", disabled=True)
        if active_unit is not None:
            if st.button("Mark this unit complete ✅"):
                st.session_state.unit_done[active_unit] = True
                st.rerun()

def figure_show(fig):
    st.pyplot(fig, use_container_width=True)
