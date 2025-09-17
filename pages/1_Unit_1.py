import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import app_header, real_world_box, try_this_box, quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 1 — Linear Algebra Foundations", layout="wide")
progress_sidebar(active_unit=1)

with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

app_header("UNIT 1: Linear Algebra Foundations",
           "Think of a matrix as a **machine** that transforms space — stretching, rotating, shearing.")

st.subheader("1.1 Solve Linear Systems (NumPy) + Visual Intuition")
c1, c2 = st.columns([2,3])
with c1:
    a11 = st.number_input("A[0,0]", value=3.0, step=0.5)
    a12 = st.number_input("A[0,1]", value=2.0, step=0.5)
    a21 = st.number_input("A[1,0]", value=1.0, step=0.5)
    a22 = st.number_input("A[1,1]", value=2.0, step=0.5)
    b1  = st.number_input("b[0]",   value=5.0, step=0.5)
    b2  = st.number_input("b[1]",   value=5.0, step=0.5)
    A = np.array([[a11, a12],[a21, a22]], float)
    b = np.array([b1, b2], float)
    try:
        x = np.linalg.solve(A, b)
        st.success(f"Exact solution x = {x}")
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        st.warning("Singular/ill-conditioned A → using least-squares.")
        st.info(f"Least-squares solution x ≈ {x}")
with c2:
    xs = np.linspace(-5, 5, 400)
    def line_y(a1, a2, bb):
        return (bb - a1*xs) / a2 if abs(a2) > 1e-12 else np.full_like(xs, np.nan)
    y1 = line_y(a11, a12, b1); y2 = line_y(a21, a22, b2)
    fig, ax = plt.subplots()
    ax.plot(xs, y1); ax.plot(xs, y2)
    try:
        ax.scatter([x[0]], [x[1]])
    except Exception:
        pass
    ax.set_title("Solve as Line Intersection"); ax.set_xlabel("x"); ax.set_ylabel("y")
    figure_show(fig)

real_world_box("Circuit analysis, FEM, calibration use linear systems.")
try_this_box("Make rows proportional to create a singular A; observe least-squares.")

st.markdown("---")
st.subheader("1.2 Vectors & Vector Spaces (2D)")
vx = st.slider("v1.x", -3.0, 3.0, 2.0, 0.1)
vy = st.slider("v1.y", -3.0, 3.0, 1.0, 0.1)
v2x = st.slider("v2.x", -3.0, 3.0, 1.0, 0.1)
v2y = st.slider("v2.y", -3.0, 3.0, 2.0, 0.1)
v1 = np.array([vx, vy]); v2 = np.array([v2x, v2y])
fig, ax = plt.subplots()
ax.axhline(0); ax.axvline(0)
ax.quiver(0,0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1)
ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_title("2D Vectors")
figure_show(fig)

M = np.column_stack([v1, v2])
det = np.linalg.det(M)
st.write(f"det([v1 v2]) = {det:.4f}")
st.write("Independent" if abs(det)>1e-8 else "Dependent")

st.markdown("---")
st.subheader("1.3 Matrix Transformations & Determinant (Area Scale)")
a11 = st.slider("A[0,0]", -2.0, 2.0, 1.0, 0.1)
a12 = st.slider("A[0,1]", -2.0, 2.0, 0.5, 0.1)
a21 = st.slider("A[1,0]", -2.0, 2.0, 0.2, 0.1)
a22 = st.slider("A[1,1]", -2.0, 2.0, 1.0, 0.1)
A = np.array([[a11,a12],[a21,a22]], float)
detA = np.linalg.det(A)
st.write(f"det(A) = {detA:.3f} (area scale, sign = orientation)")
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], float)
trans = square @ A.T
fig, ax = plt.subplots()
ax.plot(square[:,0], square[:,1]); ax.plot(trans[:,0], trans[:,1])
ax.set_aspect("equal","box"); ax.set_title("Matrix as Space Transformer")
figure_show(fig)

st.markdown("---")
st.subheader("1.4 Fundamental Subspaces + Rank")
A = np.array([[1,2,3],[2,4,6],[1,1,0]], float)
U,S,Vt = np.linalg.svd(A)
rank = int(np.sum(S>1e-10)); null_basis = Vt[rank:].T
st.write("Singular values:", np.round(S,4), "→ rank(A) =", rank)
st.write("Null space basis (columns):"); st.dataframe(np.round(null_basis,4))

st.markdown("---")
st.subheader("1.5 Eigenvalues/Eigenvectors — Interactive Transform")
A = np.array([[1.1,0.2],[0.0,0.9]], float)
vals, vecs = np.linalg.eig(A)
st.write("Eigenvalues:", np.round(vals,4))
k = st.slider("Power k", 0, 25, 10)
theta = np.linspace(0, 2*np.pi, 150)
pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
Pk = pts @ np.linalg.matrix_power(A, k).T
fig, ax = plt.subplots()
ax.scatter(pts[:,0], pts[:,1], s=8); ax.scatter(Pk[:,0], Pk[:,1], s=8)
for i in range(vecs.shape[1]):
    v = vecs[:, i].real
    ax.plot([0, 1.5*v[0]], [0, 1.5*v[1]], "--")
ax.set_aspect("equal","box"); ax.set_title("Eigen-directions emerge")
figure_show(fig)

st.markdown("---")
st.subheader("1.6 Least Squares (Line Fit) & Orthogonality")
n = st.slider("n points", 20, 200, 60, 5)
x = np.linspace(0, 10, n)
noise = st.slider("Noise σ", 0.0, 5.0, 2.0, 0.1)
y_true = 2.5*x + 5
y = y_true + np.random.normal(0, noise, size=n)
X = np.c_[np.ones(n), x]
beta = np.linalg.pinv(X.T @ X) @ X.T @ y
y_pred = X @ beta
fig, ax = plt.subplots()
ax.scatter(x, y, s=12); ax.plot(x, y_true); ax.plot(x, y_pred)
ax.set_title("Least Squares Fit"); figure_show(fig)
st.write("Estimated [intercept, slope]:", np.round(beta,4))

quiz_block(
    ["Geometric meaning of det(A) in 2D?",
     "When are vectors linearly dependent?",
     "What does least squares guarantee about residuals?"],
    ["Area scale factor (signed).",
     "One is a linear combination of the others.",
     "Residuals are orthogonal to column space of X."]
)
