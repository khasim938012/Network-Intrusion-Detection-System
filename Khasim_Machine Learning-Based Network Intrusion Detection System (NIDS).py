# %% [markdown]
# # IBM SkillsBuild Cyber Security 2026 - Conducted by BharatCares
# ## Project Title: Machine Learning-Based Network Intrusion Detection System (NIDS)
# **Author:** Your Name  
# **Date:** October 2026  
# 
# ### Project Overview
# This project implements an end-to-end Network Intrusion Detection System (NIDS) using Machine Learning. It analyzes network traffic packets to classify connections as either **Normal** or **Malicious (Attack)** (such as DoS, Probe, R2L, or U2R attacks).

# %% [code]
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Data Loading & Generation
# The code first attempts to load the benchmark **NSL-KDD** dataset from a public repository. If an internet connection is unavailable or the URL is unreachable, it automatically generates a realistic synthetic network intrusion dataset matching NSL-KDD features so the pipeline runs seamlessly anywhere.

# %% [code]
 COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

DATASET_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"

def load_or_generate_data():
    try:
        print("[INFO] Attempting to download NSL-KDD dataset...")
        df = pd.read_csv(DATASET_URL, names=COLUMNS)
        df.drop("difficulty", axis=1, inplace=True)
        # Take a stratified sample of 15,000 rows for fast execution
        df = df.sample(n=15000, random_state=42).reset_index(drop=True)
        print(f"[SUCCESS] Loaded NSL-KDD dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[WARNING] Could not fetch external dataset ({e}). Generating synthetic network traffic dataset...")
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            "duration": np.random.exponential(scale=10, size=n_samples).astype(int),
            "protocol_type": np.random.choice(["tcp", "udp", "icmp"], size=n_samples, p=[0.7, 0.2, 0.1]),
            "service": np.random.choice(["http", "private", "domain_u", "smtp", "ftp_data", "ssh"], size=n_samples),
            "flag": np.random.choice(["SF", "S0", "REJ", "RSTR"], size=n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            "src_bytes": np.random.lognormal(mean=5, sigma=2, size=n_samples).astype(int),
            "dst_bytes": np.random.lognormal(mean=5, sigma=2.5, size=n_samples).astype(int),
            "land": np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01]),
            "wrong_fragment": np.random.choice([0, 1, 3], size=n_samples, p=[0.95, 0.03, 0.02]),
            "urgent": np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005]),
            "hot": np.random.poisson(lam=0.2, size=n_samples),
            "num_failed_logins": np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.92, 0.04, 0.02, 0.01, 0.01]),
            "logged_in": np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
            "count": np.random.randint(1, 512, size=n_samples),
            "srv_count": np.random.randint(1, 512, size=n_samples),
            "serror_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "srv_serror_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "rerror_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "same_srv_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "diff_srv_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "dst_host_count": np.random.randint(1, 255, size=n_samples),
            "dst_host_srv_count": np.random.randint(1, 255, size=n_samples),
            "dst_host_same_srv_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
            "dst_host_serror_rate": np.round(np.random.uniform(0, 1, size=n_samples), 2),
        }
        df = pd.DataFrame(data)
        
        # Create realistic attack patterns based on network heuristics
        attack_score = (
            (df["flag"] == "S0").astype(int) * 2.5 +
            (df["serror_rate"] > 0.5).astype(int) * 2.0 +
            (df["count"] > 250).astype(int) * 1.5 +
            (df["num_failed_logins"] > 0).astype(int) * 2.0 +
            (df["logged_in"] == 0).astype(int) * 1.0 +
            np.random.normal(0, 1.0, size=n_samples)
        )
        df["label"] = np.where(attack_score > 2.8, "neptune", "normal")
        print(f"[SUCCESS] Generated synthetic dataset with shape: {df.shape}")
        return df

df = load_or_generate_data()
print(df.head())

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA) & Label Mapping
# Convert the target `label` into binary classes:
# * `0` = Normal traffic
# * `1` = Malicious / Intrusion traffic

# %% [code]
df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
df.drop("label", axis=1, inplace=True)

print("\n--- Class Distribution ---")
print(df["binary_label"].value_counts().rename(index={0: "Normal (0)", 1: "Attack (1)"}))

# Plot Class Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="binary_label", data=df, palette=["#2ecc71", "#e74c3c"])
plt.title("Network Traffic Distribution (0 = Normal, 1 = Attack)")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Connections")
plt.xticks([0, 1], ["Normal (0)", "Attack (1)"])
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

# Plot Protocol Type vs Traffic Class
if "protocol_type" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(x="protocol_type", hue="binary_label", data=df, palette="Set2")
    plt.title("Protocol Type Distribution by Traffic Class")
    plt.xlabel("Protocol Type")
    plt.ylabel("Count")
    plt.legend(title="Class", labels=["Normal", "Attack"])
    plt.tight_layout()
    plt.savefig("protocol_distribution.png", dpi=300)
    plt.show()

# %% [markdown]
# ## 3. Data Preprocessing & Feature Engineering
# Encode categorical variables (`protocol_type`, `service`, `flag`) and scale numerical features using `StandardScaler`.

# %% [code]
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical Columns to Encode: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop("binary_label", axis=1)
y = df["binary_label"]

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Set Shape: {X_train_scaled.shape}")
print(f"Testing Set Shape : {X_test_scaled.shape}")

# %% [markdown]
# ## 4. Model Training & Evaluation
# We train two models to compare performance:
# 1. **Logistic Regression** (Baseline Linear Classifier)
# 2. **Random Forest Classifier** (Ensemble Tree-based Classifier)

# %% [code]
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\n==========================================")
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        "model": model,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# %% [markdown]
# ## 5. Visualizing Results (Confusion Matrix, ROC Curve & Feature Importance)

# %% [code]
# 5.1 Confusion Matrix for Random Forest
rf_preds = results["Random Forest"]["y_pred"]
cm = confusion_matrix(y_test, rf_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)
plt.title("Confusion Matrix - Random Forest NIDS")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 5.2 ROC Curve Comparison
plt.figure(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# 5.3 Top 10 Feature Importances (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)

plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind="barh", color="#3498db")
plt.title("Top 10 Most Important Network Features for Intrusion Detection")
plt.xlabel("Feature Importance Score")
plt.ylabel("Network Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# %% [markdown]
# ## 6. Save Model & Real-Time Intrusion Prediction Demo

# %% [code]
joblib.dump(rf_model, "nids_random_forest_model.pkl")
joblib.dump(scaler, "nids_scaler.pkl")
print("[INFO] Saved trained model to 'nids_random_forest_model.pkl' and scaler to 'nids_scaler.pkl'.")

# Simulate predicting on a single incoming network packet
sample_packet = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample_packet)
prediction = rf_model.predict(sample_scaled)[0]
confidence = rf_model.predict_proba(sample_scaled)[0][prediction] * 100

status = "MALICIOUS INTRUSION DETECTED!" if prediction == 1 else "NORMAL TRAFFIC"
print(f"\n--- Live Packet Inspection Demo ---")
print(f"Prediction Status : {status}")
print(f"Confidence Score  : {confidence:.2f}%")
print(f"Actual Label      : {'Attack' if y_test.iloc[0] == 1 else 'Normal'}")