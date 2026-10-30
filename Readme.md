# Machine Learning-Based Network Intrusion Detection System (NIDS)

**Program:** IBM SkillsBuild Cyber Security 2026  
**Conducted By:** BharatCares  
**Author:** Your Name  

---

## 1. Project Overview
This project implements an automated **Network Intrusion Detection System (NIDS)** using Machine Learning in Python. By analyzing network traffic features (such as protocol type, byte counts, error rates, and connection flags), the system classifies network connections into **Normal Traffic (`0`)** or **Malicious Intrusion (`1`)**.

The project compares **Logistic Regression** and **Random Forest Classifier** models, generates diagnostic cybersecurity visualizations, and exports a production-ready model artifact for real-time packet inspection.

---

## 2. Dataset Information
* **Dataset Name:** NSL-KDD Dataset (Network Socket Layer - Knowledge Discovery in Databases)
* **Dataset Link:** [NSL-KDD Official Repository (University of New Brunswick / GitHub Mirror)](https://github.com/defcom17/NSL_KDD)
* **Direct Training Data URL:** [KDDTrain+.txt](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt)
* **Offline Fallback Support:** The project script automatically downloads the dataset during execution. If run in an offline environment without internet access, it automatically generates a realistic synthetic network flow dataset so the code executes without errors.

---

## 3. Technologies & Libraries Used
* **Programming Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest, Logistic Regression, StandardScaler, LabelEncoder)
* **Data Visualization:** `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`
* **Development Environment:** Jupyter Notebook / VS Code

---

## 4. Repository File Structure
```text
├── YourName_NetworkIntrusionDetection.ipynb / .py   # Complete Project Source Code
├── requirements.txt                                 # Python Dependencies
├── YourName_ProjectReport.docx                      # Complete Project Documentation
└── README.md                                        # Project Overview & Setup Guide
```

---

## 5. Setup & Run Instructions

### Step 1: Clone or Download the Project
Download all project files into a dedicated directory on your local machine.

### Step 2: Install Dependencies
Open a terminal or command prompt in the project directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Code
* **As a Python Script:**
  ```bash
  python YourName_NetworkIntrusionDetection.py
  ```
* **In Jupyter Notebook / VS Code:**
  Open `YourName_NetworkIntrusionDetection.ipynb` (or open the `.py` file directly in VS Code using the `# %%` interactive Jupyter cell markers) and click **Run All**.

---

## 6. Key Outputs & Highlights
Upon execution, the script automatically produces:
1. **Console Evaluation Report:** Accuracy, Precision, Recall, $F_1$-Score, and a live single-packet intrusion prediction demo.
2. **Saved Visual Charts (`.png`):**
   * `class_distribution.png`
   * `protocol_distribution.png`
   * `confusion_matrix.png`
   * `roc_curve.png`
   * `feature_importance.png`
3. **Trained Model Artifacts (`.pkl`):**
   * `nids_random_forest_model.pkl`
   * `nids_scaler.pkl`