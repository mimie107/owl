import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import seaborn as sns 
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Owl Movement Classifier", page_icon="🦉", layout="wide")

st.title("🦉 Owl Movement Classifier — XGBoost + SHAP + RAG")


# =============================================
# 1 — LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    return joblib.load("clasifier_model.pkl")

clf = load_model()


# =============================================
# 2 — FEATURES
# =============================================
FEATURES = [
    "snr", "sigsd", "noise", "burstSlop",
    "snr_lag1", "snr_lag2",
    "sigsd_lag1", "noise_lag1",
    "snr_roll3", "noise_roll3",
    "hour_sin", "hour_cos",
    "day", "month"
]


# =============================================
# 3 — DATA UPLOAD
# =============================================
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload your processed df", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset loaded!")
else:
    st.warning("Upload a processed dataset to continue.")
    st.stop()


# =============================================
# 4 — SHAP EXPLAINER
# =============================================
@st.cache_resource
def load_shap():
    return shap.TreeExplainer(clf)

explainer = load_shap()

def shap_for_row(row):
    X = row[FEATURES].values.reshape(1, -1)
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, np.ndarray):
        return shap_vals[0]
    if isinstance(shap_vals, list):
        return shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]

    return np.zeros(len(FEATURES))


# =============================================
#  SHAP  EXPLAINATION TEXT
# =============================================
def shap_text_summary(shap_vals):

    friendly_names = {
        "snr": "Signal strength (owl distance)",
        "sigsd": "Signal stability",
        "noise": "Background noise",
        "burstSlop": "Change in detection pattern",
        "snr_lag1": "Signal strength (1 step earlier)",
        "snr_lag2": "Signal strength (2 steps earlier)",
        "sigsd_lag1": "Signal stability (1 step earlier)",
        "noise_lag1": "Noise (1 step earlier)",
        "snr_roll3": "Short-term signal trend (3-point)",
        "noise_roll3": "Short-term noise trend (3-point)",
        "hour_sin": "Time of day (night/day scale)",
        "hour_cos": "Time of day (cosine scale)",
        "day": "Day of month",
        "month": "Month of year"
    }

    pairs = list(zip(FEATURES, shap_vals))
    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

    explanation = "### Key Factors That Drove This Prediction\n"

    for feat, val in top:
        name = friendly_names.get(feat, feat)

        if val > 0:
            direction = "This pushed the model **toward 'movement'**."
        else:
            direction = "This pushed the model **toward 'resident'**."

        explanation += f"- **{name}**: {direction}\n"

    explanation += """

These factors reflect changes in signal strength, noise, or timing that the model
associates with either regular local activity or early signs of bird departure.
"""

    return explanation



# =============================================
# 5 — RAG DOCUMENT STORE
# =============================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

rag_docs = {
    "movement_definition": """
Movement_class = 1 represents a long gap in detections, often interpreted as
temporary departure, roaming, or early signs of migration.
""",

    "feature_info": """
The model relies on SNR, noise, lag features, rolling averages, and
time-of-day signals to infer movement patterns.
""",

    "xgboost_info": """
The classifier uses engineered features from your pipeline to detect patterns
that indicate changes in behavior or tower detectability.
"""
}

rag_embeddings = {
    k: embedder.encode(v, convert_to_tensor=True)
    for k, v in rag_docs.items()
}

def retrieve_context(query):
    q = embedder.encode(query, convert_to_tensor=True)
    sims = {k: util.cos_sim(q, emb).item() for k, emb in rag_embeddings.items()}
    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return "\n\n".join([rag_docs[k] for k, _ in ranked[:2]])


# =============================================
# SIDEBAR NAVIGATION
# =============================================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA Insights", "🔍 Prediction Explorer", "🧠 RAG Explanation"]
)

# =============================================
# PAGE 1 — HOME
# =============================================
if page == "🏠 Home":
    st.header("Welcome")
    st.write("""
Welcome to the **Owl Movement Classifier App**.

This tool helps us understand owl activity patterns at the Beaverhill Bird Observatory (BBO)
using detection data collected from the automated radio-telemetry tower.

### 🔍 What this app does
• Analyze the detection signals collected from tagged owls and:
• Explore detection patterns  
• Predict when an owl is **moving(1) vs resident(0) **  
• Understand *why* with SHAP explanations  
• Ask ecological or modelling questions using Retrieval-Augmented Generation (RAG) system assistant 

### 🦉 Why this matters
These insights help us answer key ecological questions, such as:
- How long were owls detectable after tagging?
- When are owls most active (foraging vs flight times)?
- Do signal patterns suggest local movement or departure?
- How do signal strength, noise, and detection patterns change before migration?

### 📁 How to use the app
1. Upload your processed dataset in the sidebar.  
2. Navigate between pages:
   - **EDA Insights:** Explore patterns in detection times, SNR, noise, and class balance  
   - **Prediction Explorer:** View model predictions and SHAP explanations  
   - **RAG Explanation:** Get scientific context behind movement behavior  

This app brings together ecological knowledge and AI modeling to help us
identify migration behavior and better understand owl movement patterns over time.

 
""")


# =============================================
# PAGE 2 — EDA INSIGHTS (IMPROVED VERSION)
# =============================================
elif page == "📊 EDA Insights":

    st.header("📊 Exploratory Data Analysis — Owl Detectability & Movement")

    # -----------------------------------------------------
    # 0. Dataset Preview
    # -----------------------------------------------------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------------------------
    # 1. Movement Probability by Time of Day
    # -----------------------------------------------------
    if "movement_class" in df.columns and "hour" in df.columns:
        st.subheader("🕒 How Time of Day Influences Movement")

        hourly = df.groupby("hour")["movement_class"].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hourly.index, hourly.values, marker="o")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Movement (0–1)")
        ax.set_title("Movement Probability by Hour of Day")
        st.pyplot(fig)

        st.markdown("""
        **Insight:** Owls tend to show increased movement during certain hours 
        (typically dusk or night). Behavioral rhythms can strongly influence detection patterns.
        """)

    # -----------------------------------------------------
    # 2. Correlation Heatmap — Movement vs Features
    # -----------------------------------------------------
    st.subheader("🔥 What Features Influence Movement the Most? (Correlation Heatmap)")

    corr = df[FEATURES + ["movement_class"]].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="coolwarm", square=True,
        cbar_kws={"shrink": 0.6},
        ax=ax
    )
    ax.set_title("Correlation Between Features and Movement", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

    st.markdown("""
    **Insight:**  
    - Features related to **signal stability**, **rolling noise**, and **lag trends** 
      often correlate with movement.  
    - Lower SNR and higher noise frequently precede a movement event.
    """)

    # -----------------------------------------------------
    # 3. Detection Counts per Owl
    # -----------------------------------------------------
    if "motusTagID" in df.columns:
        st.subheader("🦉 Detection Volume per Owl")

        det_counts = df["motusTagID"].value_counts().sort_values(ascending=False)

        top_k = st.slider("Show Top N Owls:", 5, 40, 20)

        fig, ax = plt.subplots(figsize=(12, 4))
        det_counts.head(top_k).plot(kind="bar", ax=ax)
        ax.set_title(f"Top {top_k} Owls by Detection Count")
        ax.set_ylabel("Detection Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("""
        **Insight:**  
        Detection variability helps identify owls with long active periods  
        vs. owls that possibly left early (low detection count).
        """)

    # -----------------------------------------------------
    # 4. SNR & Noise Trends per Owl (Downsampled)
    # -----------------------------------------------------
    if "datetime" in df.columns and "motusTagID" in df.columns:
        st.subheader("📡 SNR & Noise Over Time (Per Owl)")

        selected_owl = st.selectbox("Choose Owl:", df["motusTagID"].unique())
        owl_df = df[df["motusTagID"] == selected_owl].sort_values("datetime")

        # Downsample to avoid clutter
        plot_df = owl_df.iloc[::50, :]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(plot_df["datetime"], plot_df["snr"], label="SNR", color="blue")
        ax.plot(plot_df["datetime"], plot_df["noise"], label="Noise", color="red", alpha=0.6)
        ax.set_title(f"SNR & Noise Over Time — Owl {selected_owl}")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Insight:**  
        Movement events typically coincide with:  
        - decreasing **SNR** (owl moving farther away),  
        - spikes or instability in **noise**.
        """)

    # -----------------------------------------------------
    # 5. SNR Distribution
    # -----------------------------------------------------
    st.subheader("📊 Distribution of Signal Strength (SNR)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["snr"], bins=40, color="lightgreen", edgecolor="black")
    ax.set_title("Signal Strength (SNR) Distribution")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown("""
    **Insight:**  
    Higher SNR → owl was closer to the tower.  
    Lower SNR → owl moving outward, often preceding departure events.
    """)

    # -----------------------------------------------------
    # 6. Noise Distribution
    # -----------------------------------------------------
    st.subheader("🌫 Noise Level Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["noise"], bins=40, color="salmon", edgecolor="black")
    ax.set_title("Background Noise Distribution")
    ax.set_xlabel("Noise")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown("""
    **Insight:**  
    Environmental noise spikes can interfere with detection and often align with  
    movement transitions or weather-driven disturbances.
    """)

    # -----------------------------------------------------
    # 7. Movement Probability Trend (Model Output)
    # -----------------------------------------------------
    st.subheader("📈 Movement Probability Over Time (Model Output)")

    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        except:
            try:
                df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            except:
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

        owl_ids = df["motusTagID"].unique()
        selected_owl = st.selectbox("Choose an Owl for Model Trend:", owl_ids)

        owl_df = df[df["motusTagID"] == selected_owl].sort_values("datetime")

        X = owl_df[FEATURES].values
        movement_probs = clf.predict_proba(X)[:, 1]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(owl_df["datetime"], movement_probs, color="darkred")
        ax.axhline(0.30, linestyle="--", color="gray", label="Movement threshold (0.30)")
        ax.set_title(f"Predicted Movement Probability — Owl {selected_owl}")
        ax.set_ylabel("Model Probability")
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Insight:**  
        - Low stable values → typical resident behavior  
        - Sharp increases → early warning of departure  
        - Short spikes → temporary roaming or foraging  
        """)






# =============================================
# PAGE 3 — PREDICTION EXPLORER
# =============================================
elif page == "🔍 Prediction Explorer":

    st.header("🔍 Prediction Explorer")

    idx = st.number_input("Row index:", 0, len(df)-1, 0)
    row = df.iloc[idx]

    prob = clf.predict_proba([row[FEATURES]])[0, 1]
    pred = int(prob >= 0.30)

    st.write(f"**Predicted Class:** {'Movement (1)' if pred else 'Resident (0)'}")
    st.write(f"**Movement Probability:** {prob:.3f}")

    # SHAP
    shap_vals = shap_for_row(row)

    st.subheader("SHAP Summary")
    st.markdown(shap_text_summary(shap_vals))

    # WATERFALL
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        data=row[FEATURES].values,
        feature_names=FEATURES
    )

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_exp, show=False)
    st.pyplot(fig)

    # --------------------------------------------------------
    #  INTERPRETATION OF SHAP WATERFALL CHART
    # --------------------------------------------------------
    st.markdown("""
### 🔎 How to Read This SHAP Waterfall Chart

- **Pink bars (positive)** → push the prediction **toward Movement (1)**  
- **Blue bars (negative)** → push the prediction **toward Resident (0)**  
- **Longer bars** = stronger influence on the prediction  
""")

    friendly_names = {
        "snr": "Signal strength (owl distance)",
        "sigsd": "Signal stability",
        "noise": "Background noise",
        "burstSlop": "Change in detection pattern",
        "snr_lag1": "Signal strength (1 step earlier)",
        "snr_lag2": "Signal strength (2 steps earlier)",
        "sigsd_lag1": "Signal stability (1 step earlier)",
        "noise_lag1": "Noise (1 step earlier)",
        "snr_roll3": "Short-term signal trend",
        "noise_roll3": "Short-term noise trend",
        "hour_sin": "Time of day (night/day)",
        "hour_cos": "Time of day (smooth cycle)",
        "day": "Day",
        "month": "Month"
    }

    st.markdown("### 📝  Breakdown")

    for feat, val in zip(FEATURES, shap_vals):
        name = friendly_names.get(feat, feat)
        if val > 0:
            direction = "➡️ pushed toward **Movement**"
        else:
            direction = "⬅️ pushed toward **Resident**"
        st.markdown(f"- **{name}**: {direction}")



# =============================================
# PAGE 4 — RAG CHATBOT
# =============================================
elif page == "🧠 RAG Explanation":

    st.header("🧠 Owl Movement Assistant — Ask Anything")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past chat messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🦉 Assistant:** {msg['content']}")

    # User input
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Save user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Retrieve scientific/model context
        ctx = retrieve_context(user_input)

        # Compose assistant reply
        reply = f"""
### 🦉 Assistant Response

**Your question:** {user_input}

**Relevant context:**
{ctx}

Based on your question, the model’s behavior depends on patterns in signal
strength, noise, and timing that often indicate early movement or stable residency.
"""

        # Save assistant reply
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )

        st.markdown(reply)

    # Reset button
    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()


