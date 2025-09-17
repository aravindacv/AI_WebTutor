import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from utils import app_header, quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 2 — Applications + Probability", layout="wide")
progress_sidebar(active_unit=2)
with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

app_header("UNIT 2: Applications + Probability Review",
           "Arrays & broadcasting → OLS from scratch → SVD → Bayes & probability trees.")

st.subheader("2.1 NumPy Arrays: Reshape, Slice, Broadcast")
arr = np.arange(24)
st.write("Flat array:", arr)
M = arr.reshape(4, 6); st.write("Reshaped 4×6:"); st.dataframe(M)
st.write("Slice rows 1:3, cols 2:5:"); st.dataframe(M[1:3, 2:5])
v = np.array([1,2,3,4,5,6], float)
st.write("Broadcast add row-wise (M + v):"); st.dataframe(M + v)

st.markdown("---")
st.subheader("2.2 Linear Regression (Matrix Form) — Derive & Code")
n = st.slider("n samples", 30, 400, 120, 10)
x = np.linspace(0, 5, n)
noise = st.slider("Noise σ", 0.0, 2.5, 1.0, 0.1)
y = 1.2 + 3.4*x + np.random.normal(0, noise, size=n)
X = np.c_[np.ones(n), x]
beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat
fig, ax = plt.subplots()
ax.scatter(x, y, s=16); ax.plot(x, y_hat)
ax.set_title("OLS via Normal Equations"); figure_show(fig)
st.write("Estimated [intercept, slope]:", np.round(beta_hat,4))

st.markdown("---")
st.subheader("2.3 SVD — Image-like Compression")
nimg = st.slider("Image size n×n", 64, 192, 128, 32)
xx = np.linspace(-1, 1, nimg)
Xg, Yg = np.meshgrid(xx, xx)
R = np.sqrt(Xg**2 + Yg**2)
img = np.exp(-5 * R**2)
img[int(nimg*0.3):int(nimg*0.7), int(nimg*0.3):int(nimg*0.7)] += 0.6
U,S,Vt = np.linalg.svd(img, full_matrices=False)
k = st.slider("Rank k", 1, min(40, nimg), 10, 1)
recon = (U[:,:k] * S[:k]) @ Vt[:k,:]
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(); ax.imshow(img, cmap="gray"); ax.axis("off"); ax.set_title("Original"); figure_show(fig)
with c2:
    fig, ax = plt.subplots(); ax.imshow(recon, cmap="gray"); ax.axis("off"); ax.set_title(f"Reconstruction (k={k})"); figure_show(fig)
energy = (S**2).cumsum()/(S**2).sum()
fig, ax = plt.subplots(); ax.plot(np.arange(1,len(energy)+1), energy); ax.set_title("Cumulative Energy vs k"); figure_show(fig)

st.markdown("---")
st.subheader("2.4 Bayes’ Theorem — Medical Test Simulation + Tree")
pD = st.slider("Prevalence p(D)", 0.0, 0.2, 0.02, 0.005)
sens = st.slider("Sensitivity P(+|D)", 0.5, 1.0, 0.95, 0.01)
spec = st.slider("Specificity P(-|¬D)", 0.5, 1.0, 0.90, 0.01)
N = 200_000
D = np.random.rand(N) < pD
Tpos = np.empty(N, bool)
Tpos[D] = (np.random.rand(D.sum()) < sens)
Tpos[~D] = (np.random.rand((~D).sum()) < (1-spec))
post_emp = D[Tpos].mean()
post_theo = sens*pD / (sens*pD + (1-spec)*(1-pD))
st.write(f"P(D|+) empirical ≈ {post_emp:.4f} | theoretical ≈ {post_theo:.4f}")
fig, ax = plt.subplots(figsize=(7,4))
ax.axis('off')
ax.text(0.05,0.8,"Start"); ax.plot([0.1,0.32],[0.8,0.9]); ax.plot([0.1,0.32],[0.8,0.7])
ax.text(0.34,0.92,f"D ({pD:.2f})"); ax.text(0.34,0.68,f"¬D ({1-pD:.2f})")
ax.plot([0.38,0.6],[0.92,0.98]); ax.plot([0.38,0.6],[0.92,0.86])
ax.text(0.62,1.00,f"+ ({sens:.2f})"); ax.text(0.62,0.84,f"- ({1-sens:.2f})")
ax.plot([0.38,0.6],[0.68,0.74]); ax.plot([0.38,0.6],[0.68,0.62])
ax.text(0.62,0.76,f"+ ({1-spec:.2f})"); ax.text(0.62,0.60,f"- ({spec:.2f})")
ax.set_title("Probability Tree — Medical Test"); figure_show(fig)

quiz_block(
    ["Why use pinv in OLS?", "What does k control in SVD?", "When can P(D|+) be small?"],
    ["Stability and singular/ill-conditioned safety.",
     "Latent complexity retained in reconstruction.",
     "Low prevalence → false positives dominate."]
)
