import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from utils import app_header, quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 4 — Data Wrangling + EDA + PCA", layout="wide")
progress_sidebar(active_unit=4)
with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

app_header("UNIT 4: Data Wrangling + EDA + PCA",
           "Generate mock data → clean → transform → relational joins → EDA → PCA.")

st.subheader("4.1 Generate Fake Sales/Customer Data")
n_customers = st.slider("Customers", 50, 1000, 300, 50)
n_orders = st.slider("Orders", 200, 5000, 1000, 100)
customers = pd.DataFrame({
    "customer_id": np.arange(1, n_customers+1),
    "age": np.random.randint(18, 70, size=n_customers),
    "gender": np.random.choice(["Male","Female"], size=n_customers),
    "city": np.random.choice(["Metro","Town","Rural"], size=n_customers)
})
orders = pd.DataFrame({
    "order_id": np.arange(1, n_orders+1),
    "customer_id": np.random.choice(customers["customer_id"], size=n_orders),
    "amount": np.abs(np.random.normal(100, 30, size=n_orders))
})
if n_orders >= 20:
    orders.loc[np.random.choice(n_orders, min(10, n_orders//10), replace=False), "amount"] = np.nan
    orders.loc[np.random.choice(n_orders, min(5, n_orders//20), replace=False), "amount"] *= 10
orders = pd.concat([orders, orders.iloc[:min(5, len(orders))]], ignore_index=True)
st.dataframe(customers.head()); st.dataframe(orders.head())

st.subheader("4.2 Cleaning — Missing, Outliers, Duplicates")
before = len(orders)
orders = orders.drop_duplicates(subset=["order_id","customer_id","amount"])
removed = before - len(orders); st.write("Removed duplicates:", removed)
median_amount = float(orders["amount"].median())
orders["amount"] = orders["amount"].fillna(median_amount)
low, high = orders["amount"].quantile([0.01,0.99])
orders["amount"] = orders["amount"].clip(low, high)
df = orders.merge(customers, on="customer_id", how="left")
st.dataframe(df.head())

st.subheader("4.3 Transform — Encode & Scale; Simple Relational Join")
num_features = ["age","amount"]; cat_features=["gender","city"]
pre = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop='first'), cat_features)
])
X = df[num_features + cat_features]
X_proc = pre.fit_transform(X); st.write("Processed feature matrix shape:", X_proc.shape)

students = pd.DataFrame({"student_id":[1,2,3,4], "name":["Ana","Ben","Chen","Dee"]})
courses = pd.DataFrame({"course_id":[101,102,103], "title":["Linear Algebra","Probability","Data Science"]})
enrollments = pd.DataFrame({"student_id":[1,1,2,3,4,4], "course_id":[101,103,102,101,102,103]})
enr = enrollments.merge(students, on="student_id").merge(courses, on="course_id")
st.dataframe(enr)

st.subheader("4.4 EDA — Histograms, Scatter, Boxplots, Correlation")
fig, ax = plt.subplots(); ax.hist(df["amount"], bins=30); ax.set_title("Order Amount Distribution"); figure_show(fig)
fig, ax = plt.subplots(); ax.scatter(df["age"], df["amount"], s=8); ax.set_title("Age vs Amount"); ax.set_xlabel("age"); ax.set_ylabel("amount"); figure_show(fig)
groups = [df.loc[df['gender']==g, 'amount'].values for g in ['Male','Female']]
fig, ax = plt.subplots(); ax.boxplot(groups, labels=['Male','Female']); ax.set_title("Amount by Gender"); figure_show(fig)
corr = df[["age","amount"]].corr(); st.dataframe(corr)
fig, ax = plt.subplots(); im = ax.imshow(corr, interpolation='nearest'); ax.set_xticks([0,1]); ax.set_xticklabels(["age","amount"]); ax.set_yticks([0,1]); ax.set_yticklabels(["age","amount"]); ax.set_title("Correlation Heatmap"); figure_show(fig)

st.subheader("4.5 PCA — 5D → 2D")
Z = df[["amount","age"]].copy()
Z["visits"] = np.random.poisson(5, size=len(Z))
Z["score"] = 0.5*Z["amount"] + 0.1*Z["age"] + np.random.normal(0, 5, size=len(Z))
Z["promo"] = np.random.binomial(1, 0.3, size=len(Z))
Z_num = Z.astype(float).values
scaler = StandardScaler(); Z_scaled = scaler.fit_transform(Z_num)
pca = PCA(n_components=2); Z2 = pca.fit_transform(Z_scaled); expl = pca.explained_variance_ratio_
fig, ax = plt.subplots(); ax.scatter(Z2[:,0], Z2[:,1], s=10); ax.set_title("PCA: 5D → 2D"); ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}% var)"); figure_show(fig)
st.write("Explained variance ratios:", np.round(expl,4))

quiz_block(
    ["Why scale features before PCA?","What do joins achieve?","What does each successive PC maximize?"],
    ["Prevent dominance by large-scale features.","Combine info across tables via keys.","Remaining variance, orthogonal to previous PCs."]
)
