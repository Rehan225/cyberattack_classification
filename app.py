import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Cyberattack Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .threat-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #22C55E;
        font-weight: 600;
        font-size: 1.25rem;
    }
    .threat-dos {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #EF4444;
        font-weight: 600;
        font-size: 1.25rem;
    }
    .threat-probe {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #F59E0B;
        font-weight: 600;
        font-size: 1.25rem;
    }
    .threat-brute {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #F97316;
        font-weight: 600;
        font-size: 1.25rem;
    }
    .threat-other {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #A855F7;
        font-weight: 600;
        font-size: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Artifact Loader
# ==========================================
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

@st.cache_resource
def load_models_and_artifacts():
    try:
        model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
        preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
        encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        label_encoder = joblib.load(encoder_path)
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, preprocessor, label_encoder = load_models_and_artifacts()

# Feature schema extracted from the trained preprocessor
NUM_COLS = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

PROTOCOLS = ['tcp', 'udp', 'icmp']
FLAGS = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'S3', 'OTH', 'RSTOS0']
TOP_SERVICES = [
    'http', 'ftp_data', 'smtp', 'domain_u', 'ftp', 'telnet', 'eco_i', 'ecr_i',
    'private', 'other', 'auth', 'finger', 'pop_3', 'time', 'urh_i'
]

# Benchmark results from the 6 algorithms
BENCHMARK_DATA = pd.DataFrame([
    {"Algorithm": "Logistic Regression (Deployed)", "Accuracy": 0.7970, "Macro Precision": 0.7564, "Macro Recall": 0.6591, "Macro F1-score": 0.6802, "Train Time (s)": 4.54, "Inference Time (s)": 0.002},
    {"Algorithm": "K-Nearest Neighbors (KNN)", "Accuracy": 0.7552, "Macro Precision": 0.7864, "Macro Recall": 0.5749, "Macro F1-score": 0.5885, "Train Time (s)": 0.004, "Inference Time (s)": 1.491},
    {"Algorithm": "Gradient Boosting", "Accuracy": 0.7783, "Macro Precision": 0.7200, "Macro Recall": 0.5363, "Macro F1-score": 0.5508, "Train Time (s)": 64.52, "Inference Time (s)": 0.064},
    {"Algorithm": "Random Forest", "Accuracy": 0.7469, "Macro Precision": 0.7937, "Macro Recall": 0.4854, "Macro F1-score": 0.4987, "Train Time (s)": 0.46, "Inference Time (s)": 0.014},
    {"Algorithm": "Decision Tree", "Accuracy": 0.7416, "Macro Precision": 0.4911, "Macro Recall": 0.5083, "Macro F1-score": 0.4720, "Train Time (s)": 0.45, "Inference Time (s)": 0.003},
    {"Algorithm": "Naive Bayes (Gaussian)", "Accuracy": 0.5467, "Macro Precision": 0.4484, "Macro Recall": 0.4953, "Macro F1-score": 0.4292, "Train Time (s)": 0.04, "Inference Time (s)": 0.022}
])

# Quick attack scenario presets for 1-click evaluation
SCENARIOS = {
    "Select a Preset Scenario...": None,
    "Normal User Web Browsing": {
        "protocol_type": "tcp", "service": "http", "flag": "SF", "duration": 0,
        "src_bytes": 350, "dst_bytes": 2400, "logged_in": 1, "num_failed_logins": 0,
        "count": 5, "srv_count": 5, "serror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "dst_host_count": 30, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_serror_rate": 0.0
    },
    "Denial of Service (DoS - Neptune SYN Flood)": {
        "protocol_type": "tcp", "service": "private", "flag": "S0", "duration": 0,
        "src_bytes": 0, "dst_bytes": 0, "logged_in": 0, "num_failed_logins": 0,
        "count": 250, "srv_count": 12, "serror_rate": 1.0, "same_srv_rate": 0.05,
        "diff_srv_rate": 0.07, "dst_host_count": 255, "dst_host_srv_count": 10,
        "dst_host_same_srv_rate": 0.04, "dst_host_serror_rate": 1.0
    },
    "Port Reconnaissance / Host Scan (Probe)": {
        "protocol_type": "icmp", "service": "eco_i", "flag": "SF", "duration": 0,
        "src_bytes": 20, "dst_bytes": 0, "logged_in": 0, "num_failed_logins": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 0.0, "dst_host_serror_rate": 0.0
    },
    "Credential Cracking / Password Guess (Brute Force)": {
        "protocol_type": "tcp", "service": "ftp", "flag": "SF", "duration": 0,
        "src_bytes": 26, "dst_bytes": 157, "logged_in": 0, "num_failed_logins": 1,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "dst_host_count": 218, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 0.01, "dst_host_serror_rate": 0.0
    },
    "Privilege Escalation Exploit (Other Attack)": {
        "protocol_type": "tcp", "service": "ftp", "flag": "SF", "duration": 8,
        "src_bytes": 220, "dst_bytes": 688, "logged_in": 0, "num_failed_logins": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "dst_host_count": 1, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 1.0, "dst_host_serror_rate": 0.0
    }
}

# ==========================================
# 3. Sidebar Information
# ==========================================
with st.sidebar:
    # st.image("https://img.icons8.com/fluency/96/shield.png", width=72)
    st.title("Cyberattack Classifier")
    st.markdown("**University Machine Learning Project**")
    st.caption("Benchmark: NSL-KDD Network Telemetry")
    
    st.divider()
    st.markdown("### Active Model")
    st.info(f"**{type(model).__name__}**\n- Accuracy: **79.70%**\n- Macro F1: **0.6802**\n- Trained with: SMOTE Balancing")
    
    st.divider()
    st.markdown("### Threat Categories")
    st.markdown("""
    - **Normal**: Benign network traffic
    - **DoS**: Flooding / Resource exhaustion
    - **Probe**: Port scanning & network mapping
    - **Brute Force**: Password & credential cracking
    - **Other Attack**: U2R Privilege escalation / Rootkit
    """)
    st.caption("Developed for Automated SOC Incident Response")

# ==========================================
# 4. Header Section
# ==========================================
st.markdown('<div class="main-title">Cyberattack Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Class Machine Learning System for Network Threat Detection & Incident Response</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Real-Time Threat Prediction",
    "Comparative Study (6 Models)",
    "Dataset & Feature Importance",
    # "Case Study Analysis & Q&A"
])

# ==========================================
# TAB 1: Real-Time Prediction
# ==========================================
with tab1:
    st.markdown("### Network Connection Inspection")
    st.markdown("Provide network telemetry attributes below or select a predefined threat scenario to predict the cyberattack category in real time.")
    
    # Preset Selector
    selected_scenario_name = st.selectbox(
        "Quick Test Presets (Load Pre-configured Traffic Signatures):",
        options=list(SCENARIOS.keys())
    )
    
    preset = SCENARIOS.get(selected_scenario_name) or {}
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Connection Basics")
            proto_val = preset.get("protocol_type", "tcp")
            protocol_type = st.selectbox("Protocol Type", PROTOCOLS, index=PROTOCOLS.index(proto_val) if proto_val in PROTOCOLS else 0)
            
            srv_val = preset.get("service", "http")
            service = st.selectbox("Network Service", TOP_SERVICES, index=TOP_SERVICES.index(srv_val) if srv_val in TOP_SERVICES else 0)
            
            flag_val = preset.get("flag", "SF")
            flag = st.selectbox("Connection Flag", FLAGS, index=FLAGS.index(flag_val) if flag_val in FLAGS else 0,
                               help="SF: Normal connection | S0: SYN flood without response | REJ: Connection rejected")
            
            duration = st.number_input("Connection Duration (sec)", min_value=0, max_value=60000, value=int(preset.get("duration", 0)))
        
        with col2:
            st.markdown("#### 2. Traffic Volume & Authentication")
            src_bytes = st.number_input("Source to Destination Bytes", min_value=0, max_value=10000000, value=int(preset.get("src_bytes", 350)))
            dst_bytes = st.number_input("Destination to Source Bytes", min_value=0, max_value=10000000, value=int(preset.get("dst_bytes", 2400)))
            
            logged_in = st.selectbox("User Logged In Successfully?", options=[0, 1], index=int(preset.get("logged_in", 1)),
                                     format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            
            num_failed_logins = st.number_input("Failed Login Attempts", min_value=0, max_value=10, value=int(preset.get("num_failed_logins", 0)))
            
        with col3:
            st.markdown("#### 3. Traffic Density & Rate Metrics")
            count = st.number_input("Connections to Same Host (past 2s)", min_value=0, max_value=512, value=int(preset.get("count", 5)))
            srv_count = st.number_input("Connections to Same Service (past 2s)", min_value=0, max_value=512, value=int(preset.get("srv_count", 5)))
            serror_rate = st.slider("SYN Error Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=float(preset.get("serror_rate", 0.0)), step=0.05)
            same_srv_rate = st.slider("Same Service Connection Rate", min_value=0.0, max_value=1.0, value=float(preset.get("same_srv_rate", 1.0)), step=0.05)
            dst_host_count = st.number_input("Destination Host Count", min_value=0, max_value=255, value=int(preset.get("dst_host_count", 30)))
            dst_host_srv_count = st.number_input("Destination Host Service Count", min_value=0, max_value=255, value=int(preset.get("dst_host_srv_count", 255)))

        submit_btn = st.form_submit_button("Classify Network Activity", type="primary", use_container_width=True)
        
    if submit_btn:
        if model is None or preprocessor is None:
            st.error("Model artifacts not loaded. Please ensure model_artifacts/ exists.")
        else:
            # Construct complete feature row
            input_dict = {col: [0.0] for col in NUM_COLS}
            input_dict['duration'] = [float(duration)]
            input_dict['src_bytes'] = [float(src_bytes)]
            input_dict['dst_bytes'] = [float(dst_bytes)]
            input_dict['logged_in'] = [float(logged_in)]
            input_dict['num_failed_logins'] = [float(num_failed_logins)]
            input_dict['count'] = [float(count)]
            input_dict['srv_count'] = [float(srv_count)]
            input_dict['serror_rate'] = [float(serror_rate)]
            input_dict['srv_serror_rate'] = [float(serror_rate)]
            input_dict['same_srv_rate'] = [float(same_srv_rate)]
            input_dict['dst_host_count'] = [float(dst_host_count)]
            input_dict['dst_host_srv_count'] = [float(dst_host_srv_count)]
            input_dict['dst_host_same_srv_rate'] = [float(preset.get("dst_host_same_srv_rate", 1.0))]
            input_dict['dst_host_serror_rate'] = [float(preset.get("dst_host_serror_rate", serror_rate))]
            input_dict['protocol_type'] = [protocol_type]
            input_dict['service'] = [service]
            input_dict['flag'] = [flag]
            
            df_input = pd.DataFrame(input_dict)
            
            # Predict
            X_transformed = preprocessor.transform(df_input)
            prediction_idx = model.predict(X_transformed)[0]
            predicted_category = label_encoder.inverse_transform([prediction_idx])[0]
            
            # Confidence probabilities if available
            has_proba = hasattr(model, "predict_proba")
            probs = model.predict_proba(X_transformed)[0] if has_proba else None
            
            st.divider()
            st.markdown("### Classification Result:")
            
            # Formatted Output Banner
            threat_styles = {
                "Normal": ("threat-normal", "NORMAL TRAFFIC (BENIGN)", "Traffic conforms to expected baseline behavior. No defensive action required."),
                "DoS": ("threat-dos", "DENIAL OF SERVICE (DoS) ATTACK", "High-volume connection flood detected. Mitigation: Enable IP rate-limiting and drop incomplete SYN requests."),
                "Probe": ("threat-probe", "RECONNAISSANCE / PROBE DETECTED", "Host or port scanning detected. Mitigation: Block scanning source IP and inspect firewall ingress filters."),
                "Brute Force": ("threat-brute", "BRUTE FORCE CREDENTIAL ATTACK", "Repeated authentication failures detected. Mitigation: Lock account, enforce multi-factor authentication and fail2ban rules."),
                "Other Attack": ("threat-other", "PRIVILEGE ESCALATION / EXPLOIT (U2R)", "Attempted unauthorized root shell or malicious exploit. Mitigation: Isolate compromised host immediately.")
            }
            
            css_class, title_text, playbook_text = threat_styles.get(
                predicted_category, ("threat-normal", f"PREDICTED: {predicted_category}", "Monitor connection.")
            )
            
            st.markdown(f'<div class="{css_class}">{title_text}</div>', unsafe_allow_html=True)
            st.markdown(f"**Security Playbook Guidance:** {playbook_text}")
            
            # Probabilities bar chart
            if has_proba:
                st.markdown("#### Threat Class Probabilities:")
                prob_df = pd.DataFrame({
                    "Attack Category": label_encoder.classes_,
                    "Confidence (%)": (probs * 100).round(2)
                }).sort_values(by="Confidence (%)", ascending=False)
                
                st.bar_chart(prob_df.set_index("Attack Category"), color="#1E3A8A")

# ==========================================
# TAB 2: Comparative Study
# ==========================================
with tab2:
    st.markdown("### Comparative Performance: 6 Machine Learning Algorithms")
    st.markdown("All models were trained on NSL-KDD benchmark training data with **SMOTE class balancing** and rigorously tested on the independent **KDDTest+** test set containing novel zero-day attack variants.")
    
    # Metrics Table
    st.dataframe(
        BENCHMARK_DATA.style.format({
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1-score": "{:.4f}",
            "Train Time (s)": "{:.2f}",
            "Inference Time (s)": "{:.3f}"
        }).highlight_max(subset=['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-score'], color='#DCFCE7'),
        use_container_width=True
    )
    
    st.markdown("#### Performance Metric Benchmark:")
    chart_metric = st.selectbox("Select Metric to Visualize:", ["Macro F1-score", "Accuracy", "Macro Precision", "Macro Recall"])
    chart_df = BENCHMARK_DATA[["Algorithm", chart_metric]].set_index("Algorithm")
    st.bar_chart(chart_df, color="#2563EB")
    
    st.divider()
    st.markdown("### Key Observations from the Comparative Study:")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - **Why Macro F1 is the Deciding Metric:** Accuracy is misleading in cybersecurity due to heavy class imbalance (Normal and DoS represent >85% of traffic). **Macro F1-score** gives equal weight to minority attacks like Brute Force and Privilege Escalation.
        - **Logistic Regression Won on Macro F1 (0.6802):** Linear hyperplanes fit with balanced weights achieved the best trade-off between sensitivity on minority attacks without excessively degrading majority classes.
        """)
    with col_b:
        st.markdown("""
        - **KNN & Tree Ensembles:** K-Nearest Neighbors delivered strong precision (78.6%) and a 0.5885 Macro F1, but exhibited higher inference latency (~1.49s). Random Forest and Gradient Boosting excelled at detecting known attacks but suffered higher false-positive rates on zero-day variants.
        - **Naive Bayes Limitation:** Gaussian NB performed lowest (Macro F1: 0.4292) because continuous network features violate the strong conditional independence assumption.
        """)

# ==========================================
# TAB 3: Dataset & Feature Importance
# ==========================================
with tab3:
    st.markdown("### NSL-KDD Benchmark Dataset & Feature Importance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 5-Class Target Distribution (Test Set):")
        dist_data = pd.DataFrame({
            "Threat Category": ["Normal", "DoS", "Brute Force", "Probe", "Other Attack"],
            "Test Set Count": [9711, 7458, 2885, 2421, 69],
            "Percentage (%)": [43.08, 33.08, 12.80, 10.74, 0.31]
        })
        st.dataframe(dist_data, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Zero-Day Generalization in Test Set:")
        st.info("""
        The NSL-KDD test set (**KDDTest+**) deliberately includes **18 novel attack types** that never appeared in the training set (e.g., `apache2`, `saint`, `mscan`, `mailbomb`, `udpstorm`, `sqlattack`). This tests whether ML models memorize signatures or learn generalizable threat behaviors.
        """)
        
    st.divider()
    st.markdown("#### Top Distinguishing Network Telemetry Features (Random Forest Gini Importance):")
    feature_imp_df = pd.DataFrame([
        {"Feature": "src_bytes (Source Payload Bytes)", "Importance": 0.1624, "Cybersecurity Interpretation": "Massive payload volumes isolate DoS floods; minimal payloads indicate probing."},
        {"Feature": "dst_bytes (Destination Payload Bytes)", "Importance": 0.1189, "Cybersecurity Interpretation": "Distinguishes interactive sessions from automated scanning scripts."},
        {"Feature": "serror_rate (SYN Error Rate)", "Importance": 0.0892, "Cybersecurity Interpretation": "Incomplete handshakes are the primary signature of SYN-flood DoS (Neptune)."},
        {"Feature": "same_srv_rate (Same Service Rate)", "Importance": 0.0751, "Cybersecurity Interpretation": "Consistency across ports; low values indicate diverse port scanning (Probes)."},
        {"Feature": "dst_host_srv_count (Host Service Count)", "Importance": 0.0684, "Cybersecurity Interpretation": "Number of connections to the same service; separates benign servers from targets."},
        {"Feature": "flag_SF (Normal Established Flag)", "Importance": 0.0543, "Cybersecurity Interpretation": "Differentiates properly completed sessions from aborted connections."},
        {"Feature": "num_failed_logins (Failed Logins)", "Importance": 0.0412, "Cybersecurity Interpretation": "Direct indicator of brute-force dictionary attacks against FTP/SSH/Telnet."}
    ])
    st.dataframe(feature_imp_df, hide_index=True, use_container_width=True)

# Footer
st.divider()
st.caption("Cyberattack Classification Using Machine Learning | Built with Streamlit, Scikit-Learn & NSL-KDD")
