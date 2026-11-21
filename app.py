
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM UI STYLING
# ==========================================

st.markdown("""
<style>

/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
    color: #1E293B;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Typography */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #172554 !important;
    font-weight: 700 !important;
}

p, label, span, div {
    font-family: 'Inter', sans-serif;
}

h1 {
    letter-spacing: -1.2px;
}

[data-testid="stMetricValue"],
input {
    font-variant-numeric: tabular-nums;
}

/* Input labels */
.stNumberInput label {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Number input fields */
.stNumberInput input {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    padding: 0.65rem !important;
    color: #0F172A !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Input focus */
.stNumberInput input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
}

/* Primary button */
.stButton button,
.stFormSubmitButton button {
    background-color: #4338CA !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #3730A3 !important;
    box-shadow: 0 4px 12px rgba(67, 56, 202, 0.2);
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem;
}

/* Prediction result cards */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}

.result-high {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-top: 1rem;
}

.result-low {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 1.25rem;
    margin-top: 1rem;
}

.result-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.4rem 0;
    font-variant-numeric: tabular-nums;
}

.muted-text {
    color: #475569;
    font-size: 0.85rem;
}

/* Model confidence label */
[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Model confidence percentage */
[data-testid="stMetricValue"] {
    color: #172554 !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums;
}

.section-label {
    color: #64748B;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="font-size: 2rem; margin-bottom: 0.3rem;">
        Student Performance Predictor
    </h1>
    <p style="color: #64748B; font-size: 0.95rem;">
        Analyze student learning activity and generate
        a predicted performance rating.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Prototype version: The current model predicts High or Low ratings, "
    "not Pass or Fail."
)


# ==========================================
# 5. MAIN DASHBOARD LAYOUT
# ==========================================

left_column, right_column = st.columns(
    [1.35, 0.85],
    gap="large"
)


# ==========================================
# 6. STUDENT INPUT FORM
# ==========================================

with left_column:

    with st.container(border=True):

        st.markdown("### Student Information")

        st.markdown(
            '<p class="muted-text">Enter the student activity measurements '
            'below to generate a prediction.</p>',
            unsafe_allow_html=True
        )

        with st.form("prediction_form"):

            st.markdown("##### Learning Activities")

            col1, col2 = st.columns(2)

            with col1:
                submission = st.number_input(
                    "Submission Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of submission-related activities."
                )

                quiz = st.number_input(
                    "Quiz Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of quiz-related activities."
                )

                forum = st.number_input(
                    "Forum Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of forum-related activities."
                )

                create = st.number_input(
                    "Create Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of creation activities."
                )

            with col2:
                read = st.number_input(
                    "Read Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of reading activities."
                )

                update = st.number_input(
                    "Update Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of update activities."
                )

                delete = st.number_input(
                    "Delete Actions",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of deletion activities."
                )

                days = st.number_input(
                    "Days Logged In",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Number of days the student logged into the platform."
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                "Generate Prediction",
                use_container_width=True
            )


# ==========================================
# 7. MAKE PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame([{
        "Submission.Actions": submission,
        "Quiz.Actions": quiz,
        "Forum.Actions": forum,
        "Create.Actions": create,
        "Read.Actions": read,
        "Update.Actions": update,
        "Delete.Actions": delete,
        "Days.Logged.In": days
    }])

    prediction = model.predict(input_data)[0]

    # Get predicted class probability
    probabilities = model.predict_proba(input_data)[0]

    classifier = model.named_steps["classifier"]
    predicted_index = list(classifier.classes_).index(prediction)

    confidence = probabilities[predicted_index] * 100

    # Store result for display
    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = confidence


# ==========================================
# 8. PREDICTION RESULTS PANEL
# ==========================================

with right_column:

    with st.container(border=True):

        st.markdown("### Prediction Results")

        st.markdown(
            '<p class="muted-text">Your model output will appear here.</p>',
            unsafe_allow_html=True
        )

        if "prediction" in st.session_state:

            prediction = st.session_state["prediction"]
            confidence = st.session_state["confidence"]

            if prediction == "High":

                st.markdown("""
                <div class="result-high">
                    <div class="section-label" style="color:#15803D;">
                        PREDICTED RATING
                    </div>
                    <div class="result-value" style="color:#15803D;">
                        HIGH
                    </div>
                    <div style="color:#166534; font-size:0.85rem;">
                        The model classified this student's
                        activity rating as High.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-low">
                    <div class="section-label" style="color:#B45309;">
                        PREDICTED RATING
                    </div>
                    <div class="result-value" style="color:#B45309;">
                        LOW
                    </div>
                    <div style="color:#92400E; font-size:0.85rem;">
                        The model classified this student's
                        activity rating as Low.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence represents the predicted class probability "
                "from the Random Forest model. It is not a guarantee "
                "of prediction correctness."
            )

        else:

            st.markdown("""
            <div style="
                background:#F8FAFC;
                border:1px dashed #CBD5E1;
                border-radius:12px;
                padding:2rem 1rem;
                text-align:center;
                margin-top:1rem;
            ">
                <div style="
                    font-weight:600;
                    color:#334155;
                    margin-top:0.6rem;
                ">
                    No prediction yet
                </div>
                <div style="
                    color:#64748B;
                    font-size:0.83rem;
                    margin-top:0.4rem;
                ">
                    Complete the student information form
                    and click Generate Prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    border-top:1px solid #E2E8F0;
    padding-top:1.2rem;
    color:#94A3B8;
    font-size:0.78rem;
">
    Student Performance Predictor |
    Powered by Random Forest & Streamlit
    <br>
    Machine Problem 3
</div>
""", unsafe_allow_html=True)
