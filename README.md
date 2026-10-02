# 🛡️ Cyberattack Classification Using Machine Learning

An end-to-end Machine Learning case study and comparative benchmark system for multi-class cyber threat detection on network traffic telemetry, featuring model training, class balancing, zero-day generalization analysis, and an interactive Streamlit deployment application.

---

## 📌 Project Overview & Objectives

Organizations face diverse network attacks ranging from high-volume Denial-of-Service (DoS) floods to stealthy port scanning, brute-force credential stuffing, and unauthorized privilege escalation. This project automates the identification and classification of network events into 5 primary operational categories:

1. **Normal**: Benign network traffic
2. **DoS (Denial of Service)**: Resource exhaustion attacks (e.g., `neptune`, `smurf`, `back`, `apache2`)
3. **Probe**: Network mapping and port-scanning reconnaissance (e.g., `satan`, `ipsweep`, `portsweep`, `nmap`)
4. **Brute Force**: Password guessing and remote credential attacks (e.g., `guess_passwd`, `ftp_write`, `snmpguess`)
5. **Other Attack**: Privilege escalation (U2R) and system exploits (e.g., `buffer_overflow`, `rootkit`, `sqlattack`)

---

## 🔬 Machine Learning Algorithms Evaluated

Six fundamental machine learning algorithms were trained and compared under identical conditions:
1. **Logistic Regression** (Multi-class with balanced class priors)
2. **K-Nearest Neighbors (KNN)** (Instance-based distance classifier)
3. **Decision Tree** (Interpretable axis-aligned partitioning)
4. **Random Forest** (Ensemble bagging of decorrelated decision trees)
5. **Naive Bayes** (Gaussian probabilistic baseline)
6. **Gradient Boosting** (Sequential gradient boosting ensemble)

---

## 📊 Comparative Performance Results (KDDTest+ Benchmark)

All models were evaluated on the independent official test set (**KDDTest+**, 22,544 samples) containing **18 novel zero-day attack types** that never appeared in the training set:

| Algorithm | Accuracy | Macro Precision | Macro Recall | Macro F1-score | Train Time | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Deployed)** 🏆 | **0.7970** | **0.7564** | **0.6591** | **0.6802** | 4.54s | 0.002s |
| **K-Nearest Neighbors (KNN)** | 0.7552 | 0.7864 | 0.5749 | 0.5885 | 0.004s | 1.491s |
| **Gradient Boosting** | 0.7783 | 0.7200 | 0.5363 | 0.5508 | 64.52s | 0.064s |
| **Random Forest** | 0.7469 | 0.7937 | 0.4854 | 0.4987 | 0.46s | 0.014s |
| **Decision Tree** | 0.7416 | 0.4911 | 0.5083 | 0.4720 | 0.45s | 0.003s |
| **Naive Bayes** | 0.5467 | 0.4484 | 0.4953 | 0.4292 | 0.04s | 0.022s |

### Key Academic Findings:
* **Metric Importance**: In imbalanced cyber threat detection, raw Accuracy is misleading. **Macro F1-score** is the primary evaluation metric because it gives equal weight to rare threats (Brute Force, U2R).
* **Class Balancing (SMOTE)**: Solved minority class starvation, dramatically raising Macro F1 and Recall on rare attacks.
* **Feature Importance**: Payload bytes (`src_bytes`, `dst_bytes`), SYN error rates (`serror_rate`, `flag_S0`), and connection counts (`count`) are the strongest indicators of cyberattacks.

---

## 🗂️ Project Directory Structure

```text
├── cyberattack_classification.ipynb  # End-to-end Jupyter Notebook with all outputs & plots
├── app.py                            # Interactive Streamlit deployment application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── .streamlit/
│   └── config.toml                   # Streamlit server configuration
├── model_artifacts/                  # Serialized production pipeline
│   ├── best_model.joblib             # Deployed trained model
│   ├── preprocessor.joblib           # Fitted ColumnTransformer (Scaler + OneHotEncoder)
│   └── label_encoder.joblib          # Target class label encoder
└── NSL-KDD/                          # Benchmark network intrusion dataset files
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rehan225/cyberattack_classification.git
cd cyberattack_classification
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test real-time predictions, explore 1-click threat presets, inspect feature importances, and view algorithm benchmarks.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook cyberattack_classification.ipynb
```
