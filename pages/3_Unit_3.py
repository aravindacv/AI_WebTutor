import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 3 — Distributions + Simulation", layout="wide")
progress_sidebar(active_unit=3)
with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

st.title("🎲 Unit 3: Probability Distributions + Simulation (Py3.13-ready)")

tab1, tab2, tab3, tab4 = st.tabs(["Discrete", "Continuous", "Expectation/Variance", "Simulations"])

with tab1:
    st.subheader("Discrete distributions: Bernoulli, Binomial, Poisson")
    choice = st.selectbox("Pick a distribution", ["Bernoulli", "Binomial", "Poisson"])
    if choice == "Bernoulli":
        p = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
        x = np.array([0,1])
        pmf = stats.bernoulli.pmf(x, p)
        fig, ax = plt.subplots()
        ax.bar(x, pmf, width=0.4)
        ax.set_xticks([0,1]); ax.set_xlabel("x"); ax.set_ylabel("PMF")
        ax.set_title(f"Bernoulli(p={p:.2f})")
        for xi, yi in zip(x, pmf):
            ax.text(xi, yi+0.01, f"{yi:.2f}", ha="center")
        figure_show(fig)

    elif choice == "Binomial":
        n = st.slider("n (trials)", 1, 100, 10, 1)
        p = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
        k = np.arange(0, n+1)
        pmf = stats.binom.pmf(k, n, p)
        fig, ax = plt.subplots()
        ax.bar(k, pmf, width=0.8)
        ax.set_xlabel("k"); ax.set_ylabel("PMF")
        ax.set_title(f"Binomial(n={n}, p={p:.2f})")
        figure_show(fig)

    else:  # Poisson
        lam = st.slider("λ (rate)", 0.1, 20.0, 3.0, 0.1)
        kmax = int(max(20, lam*5))
        k = np.arange(0, kmax+1)
        pmf = stats.poisson.pmf(k, lam)
        fig, ax = plt.subplots()
        ax.bar(k, pmf, width=0.8)
        ax.set_xlabel("k"); ax.set_ylabel("PMF")
        ax.set_title(f"Poisson(λ={lam:.1f})")
        figure_show(fig)

with tab2:
    st.subheader("Continuous distributions: Uniform, Normal, Exponential")
    choice = st.selectbox("Pick a distribution", ["Uniform", "Normal", "Exponential"])
    xs = np.linspace(-5, 5, 400)

    if choice == "Uniform":
        a = st.slider("a", -5.0, 4.0, -1.0, 0.1)
        b = st.slider("b", a+0.1, 5.0, 1.0, 0.1)
        dist = stats.uniform(loc=a, scale=b-a)
    elif choice == "Normal":
        mu = st.slider("μ", -2.0, 2.0, 0.0, 0.1)
        sigma = st.slider("σ", 0.1, 3.0, 1.0, 0.1)
        dist = stats.norm(loc=mu, scale=sigma)
    else:
        lam = st.slider("λ", 0.1, 3.0, 1.0, 0.1)
        dist = stats.expon(scale=1/lam)
        xs = np.linspace(0, 10, 400)

    show_cdf = st.checkbox("Show CDF", value=False)
    fig, ax = plt.subplots()
    ax.plot(xs, dist.pdf(xs), lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("PDF")
    title = f"{choice} — PDF"
    if show_cdf:
        ax2 = ax.twinx()
        ax2.plot(xs, dist.cdf(xs), lw=2, ls="--")
        ax2.set_ylabel("CDF")
        title += " & CDF"
    ax.set_title(title)
    figure_show(fig)

with tab3:
    st.subheader("Expectation & Variance via simulation")
    n = st.slider("Sample size", 100, 100_000, 5_000, 100)
    mu = st.slider("True mean μ", -1.0, 1.0, 0.0, 0.1)
    sigma = st.slider("True σ", 0.1, 3.0, 1.0, 0.1)
    rng = np.random.default_rng(0)
    x = rng.normal(mu, sigma, size=n)
    st.write(f"Sample mean ≈ {x.mean():.3f}, Sample variance ≈ {x.var(ddof=1):.3f}")
    fig, ax = plt.subplots()
    ax.hist(x, bins=40, density=True, alpha=0.8)
    grid = np.linspace(mu-4*sigma, mu+4*sigma, 400)
    ax.plot(grid, stats.norm(mu, sigma).pdf(grid), lw=2)
    ax.set_title("Histogram with true Normal PDF")
    figure_show(fig)

with tab4:
    st.subheader("Monte Carlo: π & CLT")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Estimate π by random dart throws**")
        N = st.slider("Number of darts", 100, 100000, 5000, 100)
        rng = np.random.default_rng(1)
        x = rng.random(N); y = rng.random(N)
        inside = (x**2 + y**2) <= 1.0
        pi_est = 4 * inside.mean()
        st.metric("π estimate", f"{pi_est:.5f}")
        fig, ax = plt.subplots()
        ax.scatter(x[~inside], y[~inside], s=2, alpha=0.4)
        ax.scatter(x[inside], y[inside], s=2, alpha=0.8)
        ax.set_aspect("equal"); ax.set_title("Darts in unit square")
        figure_show(fig)

    with col2:
        st.markdown("**Central Limit Theorem (sum of uniforms)**")
        m = st.slider("Sum of m uniforms", 1, 30, 10, 1)
        N2 = st.slider("Number of experiments", 500, 50000, 10000, 500)
        rng = np.random.default_rng(2)
        sums = rng.random((N2, m)).sum(axis=1)
        sums = (sums - sums.mean())/sums.std()  # standardize
        fig, ax = plt.subplots()
        ax.hist(sums, bins=60, density=True, alpha=0.8)
        xs = np.linspace(-4, 4, 400)
        ax.plot(xs, stats.norm(0,1).pdf(xs), lw=2)
        ax.set_title("CLT: standardized sums → N(0,1)")
        figure_show(fig)

quiz_block(
    ["PMF vs PDF?", "What does CLT say?", "Effect of λ on Poisson?"],
    ["PMF is discrete; PDF is continuous (integrates to 1).",
     "Means of i.i.d. samples are ~Normal for large n.",
     "Mean & variance increase with λ; mass shifts right."]
)
