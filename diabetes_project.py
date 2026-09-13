import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

charts = {}

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv("diabetes.csv", names=cols, header=None)

# Defensive: force numeric dtypes in case the source file has a stray header
# row or any non-numeric text, which would otherwise silently break zero/NaN
# detection later on.
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(how="all")  # drop any fully-unparseable rows (e.g. a header row)
raw_shape = df.shape

# ============ PART 1: CLEANING ============
# Columns where 0 is biologically impossible -> treat as missing
zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
zero_counts_before = {c: (df[c]==0).sum() for c in zero_as_missing}

df_clean = df.copy()
for c in zero_as_missing:
    df_clean[c] = df_clean[c].replace(0, np.nan)

missing_before = df_clean.isnull().sum()
missing_nonzero = missing_before[missing_before > 0]

fig, ax = plt.subplots(figsize=(6,4))
if not missing_nonzero.empty:
    missing_nonzero.sort_values().plot(kind="barh", ax=ax, color="#e07a5f")
    ax.set_title("Missing Values (Zeros Treated as Missing)")
else:
    # Defensive fallback: shouldn't happen with this dataset, but guarantees
    # the script never crashes even if an environment reads the data differently.
    ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
    ax.set_title("Missing Values (Zeros Treated as Missing)")
    ax.axis("off")
charts["missing_before"] = fig_to_b64(fig)

# Impute: median grouped by Outcome (more accurate than global median,
# since glucose/BMI patterns genuinely differ between diabetic/non-diabetic patients)
for c in zero_as_missing:
    df_clean[c] = df_clean.groupby("Outcome")[c].transform(lambda x: x.fillna(x.median()))

# Duplicates
dupes = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()

# Outliers: cap Insulin and SkinThickness (known extreme values) via IQR
outliers_capped = {}
for c in ["Insulin","SkinThickness","BMI"]:
    Q1, Q3 = df_clean[c].quantile([0.25,0.75])
    IQR = Q3-Q1
    lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
    n_out = ((df_clean[c]<lower)|(df_clean[c]>upper)).sum()
    outliers_capped[c] = n_out
    df_clean[c] = df_clean[c].clip(lower, upper)

clean_shape = df_clean.shape

# ============ PART 2: EDA ============
fig, axes = plt.subplots(2, 4, figsize=(16,7))
feature_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
for i, c in enumerate(feature_cols):
    ax = axes[i//4, i%4]
    sns.histplot(df_clean[c], bins=25, kde=True, ax=ax, color="#3d5a80")
    ax.set_title(c)
plt.tight_layout()
charts["distributions"] = fig_to_b64(fig)

# Outcome balance
fig, ax = plt.subplots(figsize=(5,4))
df_clean["Outcome"].map({0:"No Diabetes",1:"Diabetes"}).value_counts().plot(
    kind="bar", ax=ax, color=["#81b29a","#e07a5f"])
ax.set_title("Class Balance")
plt.xticks(rotation=0)
charts["class_balance"] = fig_to_b64(fig)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
charts["correlation"] = fig_to_b64(fig)

# Key features by outcome (boxplots)
fig, axes = plt.subplots(1, 3, figsize=(14,4.5))
for ax, c in zip(axes, ["Glucose","BMI","Age"]):
    sns.boxplot(data=df_clean, x="Outcome", y=c, hue="Outcome", ax=ax,
                palette=["#81b29a","#e07a5f"], legend=False)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["No Diabetes","Diabetes"])
    ax.set_title(f"{c} by Outcome")
plt.tight_layout()
charts["boxplots_outcome"] = fig_to_b64(fig)

corr_matrix = df_clean.corr()["Outcome"].drop("Outcome").sort_values(ascending=False)

# ============ PART 3: PREDICTIVE MODELING ============
X = df_clean.drop(columns=["Outcome"])
y = df_clean["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
roc_data = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    results[name] = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":roc_auc}
    roc_data[name] = (fpr, tpr, roc_auc)

    fig, ax = plt.subplots(figsize=(4,3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Diabetes","Diabetes"], yticklabels=["No Diabetes","Diabetes"])
    ax.set_title(f"{name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    charts[f"cm_{name}"] = fig_to_b64(fig)

fig, ax = plt.subplots(figsize=(6,5))
colors = {"Logistic Regression":"#3d5a80","Decision Tree":"#e07a5f","Random Forest":"#81b29a"}
for name,(fpr,tpr,roc_auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], linewidth=2)
ax.plot([0,1],[0,1],"--",color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend(loc="lower right")
charts["roc_combined"] = fig_to_b64(fig)

fig, ax = plt.subplots(figsize=(6,4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
bars = ax.bar(names, accs, color=["#3d5a80","#e07a5f","#81b29a"])
ax.set_ylim(0,1); ax.set_title("Model Accuracy Comparison")
for bar, a in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, a+0.02, f"{a:.1%}", ha="center", fontweight="bold")
charts["accuracy_comparison"] = fig_to_b64(fig)

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6,4))
importances.plot(kind="barh", ax=ax, color="#f2cc8f")
ax.set_title("Feature Importance (Random Forest)")
charts["feature_importance"] = fig_to_b64(fig)

# Save everything (in the same folder the script is run from)
df_clean.to_csv("diabetes_cleaned.csv", index=False)
with open("diabetes_results.pkl","wb") as f:
    pickle.dump({
        "charts":charts, "results":results, "zero_counts_before":zero_counts_before,
        "dupes":dupes, "outliers_capped":outliers_capped, "raw_shape":raw_shape,
        "clean_shape":clean_shape, "corr_matrix":corr_matrix
    }, f)

print("Zero counts treated as missing:", zero_counts_before)
print("Duplicates removed:", dupes)
print("Outliers capped:", outliers_capped)
print("\nModel results:")
for n,r in results.items():
    print(n, r)
best = max(results, key=lambda x: results[x]["accuracy"])
print("Best model:", best)
