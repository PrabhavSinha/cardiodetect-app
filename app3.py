import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import base64 

# --- Configuration and Custom CSS (Enhanced UI) ---
st.set_page_config(
    page_title="Cardio Predict - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea; 
        --secondary-color: #E8F6F3;
        --accent-color: #E74C3C;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --text-color: #2C3E50;
    }
    
    /* Hide streamlit branding, enforce light background for inputs */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {font-size: 3rem; margin: 0; font-weight: 700;}
    
    /* Input sections - Forced White Background */
    .input-section {
        background: white; 
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .section-title {
        color: var(--primary-color);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Results styling */
    .result-container {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    /* Prediction button style */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def render_popover(label, description):
    """Renders a Popover for definitions."""
    with st.popover(f"❓ {label}"):
        st.markdown(f'<div class="popover-content">**{label}:** {description}</div>', unsafe_allow_html=True)

def get_risk_contributions(risk_prob):
    """SIMULATED: Provides feature importance/risk contribution for the bar chart."""
    if risk_prob > 0.75:
        return {'Age': 30, 'ST Depression': 25, 'Cholesterol': 15, 'BP/HR': 10, 'Other': 20}
    elif risk_prob > 0.45:
        return {'Age': 20, 'Cholesterol': 20, 'ST Depression': 15, 'BP/HR': 15, 'Other': 30}
    else:
        return {'Age': 15, 'Cholesterol': 10, 'ST Depression': 5, 'BP/HR': 5, 'Other': 65}

def generate_report_html(user_data, risk_percentage, recommendations, contributions):
    """Generates the HTML content for the PDF/HTML download."""
    contributions_html = "".join([f"<li>{factor}: {impact}%</li>" for factor, impact in contributions.items()])
    recs_html = "".join([f"<li>{r.replace('🔸 ', '')}</li>" for r in recommendations])
    risk_color = "red" if risk_percentage >= 50 else "green"
    
    html_content = f"""
    <!DOCTYPE html><html><head><title>CardioPredict Risk Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .report-header {{ background-color: #667eea; color: white; padding: 20px; text-align: center; border-radius: 5px; }}
            h1 {{ margin-top: 0; }}
            .section {{ margin-top: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .risk-score {{ font-size: 2.5em; font-weight: bold; color: {risk_color}; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        </style>
    </head><body>
        <div class="report-header"><h1>CardioPredict Health Report</h1><p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div>
        <div class="section" style="text-align:center;"><h2>Overall Risk Assessment</h2><div class="risk-score" style="color:{risk_color};">{risk_percentage:.1f}%</div><p>Probability of Heart Disease Detected by ML Model</p></div>
        <div class="section"><h2>Patient Input Summary</h2><table>
                <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
                <tr><td>Age</td><td>{user_data.get('age', 'N/A')}</td><td>Sex</td><td>{user_data.get('sex', 'N/A')}</td></tr>
            </table></div>
        <div class="section"><h2>Top Risk Factors Contribution</h2><ul>{contributions_html}</ul></div>
        <div class="section"><h2>Personalized Recommendations</h2><ul>{recs_html}</ul></div>
        <div class="disclaimer"><h3>Disclaimer:</h3><p>This report is for informational and educational purposes only.</p></div>
    </body></html>
    """
    return html_content.encode('utf-8')

# --- Model Loading (Enabled) ---
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('LogisticRegression.pkl')
        scaler = joblib.load('scaler.pkl')
        expected_columns = joblib.load('columns.pkl')
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("Model files (LogisticRegression.pkl, scaler.pkl, columns.pkl) not found. Please place them in the same directory as app1.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        st.info("The model files might be incompatible with your current scikit-learn version. Try re-saving them or check the terminal warnings.")
        st.stop()

model, scaler, expected_columns = load_model_components()

# --- Session State Management ---
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioPredict</h1>
    <p>Advanced Heart Disease Risk Assessment Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar (UPDATED CONTENT) ---
with st.sidebar:
    st.markdown("## 📚 Resources & Learning")
    st.markdown("Understand your metrics and know where to find expert advice.")
    
    st.subheader("💬 Health Assistant (FAQ)")
    faq_question = st.selectbox("Common Questions:", ['What is LDL cholesterol?', 'What is Systolic BP?', 'What does a ST-T wave abnormality mean?'], key='faq_select')
    
    if st.button("Answer Question", key="faq_btn", use_container_width=True):
        with st.chat_message("assistant", avatar="🩺"):
            if faq_question == 'What is LDL cholesterol?': st.write("**Low-Density Lipoprotein (LDL):** Often called 'bad' cholesterol. High levels lead to plaque buildup in arteries.")
            elif faq_question == 'What is Systolic BP?': st.write("**Systolic Blood Pressure:** The top number in a BP reading, measuring pressure when your heart **beats**.")
            elif faq_question == 'What does a ST-T wave abnormality mean?': st.write("A finding on an **ECG** that can indicate poor blood supply (*ischemia*). **Requires clinical evaluation.**")

    st.markdown("---")
    
    # --- NEW AND UPDATED LINKS ---
    st.subheader("🔗 Trusted Health Links")
    
    st.markdown("### 🌍 Global Resources")
    st.markdown("- [**World Heart Federation (WHF)**](https://www.world-heart-federation.org/) - Global advocacy and information.")
    st.markdown("- [**World Health Organization (WHO)**](https://www.who.int/health-topics/cardiovascular-diseases) - Official CVD information and statistics.")
    st.markdown("- [**American Heart Association (AHA)**](https://www.heart.org/) - Leading resource for heart care and prevention.")
    
    st.markdown("### 🇮🇳 Indian Resources")
    st.markdown("- [**Indian Heart Association (IHA)**](https://indianheartassociation.org/) - Focuses on preventative cardiovascular health in India.")
    st.markdown("- [**Cardiological Society of India (CSI)**](https://www.csi.org.in/) - Leading professional body for Indian cardiologists.")
    st.markdown("- [**Indian Council of Medical Research (ICMR)**](https://www.icmr.gov.in/) - India's apex medical research body; check for CVD guidelines and studies.")
    # --- END NEW AND UPDATED LINKS ---
    
# --- Main Application Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    
    # --- Data Upload Feature ---
    with st.expander("📁 Upload Historical Data (CSV/Excel)", expanded=False):
        st.caption("Upload historical lab results to analyze trends. Columns must include: Date, RestingBP, Cholesterol.")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # --- FILE HANDLING FIX ---
                if uploaded_file.name.endswith('.csv'):
                    history_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    # Use openpyxl engine explicitly for modern Excel files
                    history_df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.warning("Unsupported file type. Please upload a CSV or XLSX file.")
                    history_df = pd.DataFrame() 
                
                if not history_df.empty:
                    st.success("File uploaded and processed successfully! Showing trends.")
                    
                    # Simple Historical Trend Visualization
                    if 'RestingBP' in history_df.columns and len(history_df) > 1:
                        st.subheader("BP Trend")
                        st.line_chart(history_df['RestingBP'])
                    if 'Cholesterol' in history_df.columns and len(history_df) > 1:
                        st.subheader("Cholesterol Trend")
                        st.line_chart(history_df['Cholesterol'])

            except ImportError:
                st.error("🚨 **Dependency Error:** Reading XLSX files requires the `openpyxl` library. Please run: `pip install openpyxl` in your terminal.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # --- Input Fields (Demographics, Vitals, Diagnostics) ---
    st.markdown("""<div class="input-section"><div class="section-title">👤 Patient Demographics</div></div>""", unsafe_allow_html=True)
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1: age = st.slider("Age", 18, 100, 40)
    with demo_col2: sex = st.selectbox("Sex", ['M', 'F'])
    
    st.markdown("""<div class="input-section"><div class="section-title">🩺 Clinical Symptoms</div></div>""", unsafe_allow_html=True)
    chest_pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    exercise_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
    
    st.markdown("""<div class="input-section"><div class="section-title">📊 Vital Signs & Lab Results</div></div>""", unsafe_allow_html=True)
    vital_col1, vital_col2 = st.columns(2)
    with vital_col1:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        render_popover("Resting BP", "Systolic pressure at rest.")
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        render_popover("Cholesterol", "Total blood cholesterol level.")
    with vital_col2:
        max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])

    st.markdown("""<div class="input-section"><div class="section-title">🔬 Diagnostic Test Results</div></div>""", unsafe_allow_html=True)
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        resting_ecg = st.selectbox("Resting ECG Results", ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        st_slope = st.selectbox("ST Segment Slope", ['Upsloping', 'Flat', 'Downsloping'])
    with diag_col2:
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        render_popover("ST Depression", "ST depression induced by exercise relative to rest.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store user inputs for report
    st.session_state.user_inputs.update({'age': age, 'sex': sex, 'trestbps': resting_bp, 'chol': cholesterol, 'thalach': max_heart_rate, 'oldpeak': oldpeak})
    
    predict_button = st.button("🔍 Analyze Heart Disease Risk", key="predict_btn")

# --- Information Panel ---
with col2:
    st.markdown('<div class="info-box"><h4>ℹ️ About This Assessment</h4><p>This advanced AI tool uses logistic regression ML to analyze multiple cardiovascular risk factors simultaneously.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><h4>📋 Risk Factors Analyzed</h4><ul><li><strong>Demographics:</strong> Age and Gender</li><li><strong>Vitals:</strong> Blood Pressure & Heart Rate</li><li><strong>Diagnostics:</strong> ECG Results & ST Depression</li></ul></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><h4>⚠️ Important Medical Notice</h4><p><strong>This is NOT a medical diagnosis.</strong> Always consult a licensed healthcare professional for medical advice.</p></div>', unsafe_allow_html=True)


# --- Prediction Logic ---
if predict_button:
    with st.spinner('Analyzing patient data...'):
        
        # Data preparation logic
        raw_data = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_blood_sugar == 'Yes' else 0, 'MaxHR': max_heart_rate,
            'Oldpeak': oldpeak, 'Sex_M': 1 if sex == 'M' else 0, 'Sex_F': 1 if sex == 'F' else 0,
            'ChestPainType_' + chest_pain: 1, 'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1, 'ST_Slope_' + st_slope: 1
        }
        input_data = pd.DataFrame([raw_data]).reindex(columns=expected_columns, fill_value=0)
        scaled_input = scaler.transform(input_data)
        
        prediction_proba = model.predict_proba(scaled_input)[0]
        risk_percentage = prediction_proba[1] * 100
        prediction = 1 if risk_percentage >= 50 else 0
        
        # Get recommendations and contributions
        recommendations = []
        if age > 55: recommendations.append("🔸 Regular cardiac check-ups recommended due to age")
        if resting_bp > 140: recommendations.append("🔸 Blood pressure management needed")
        if cholesterol > 240: recommendations.append("🔸 Cholesterol levels require attention")
        if not recommendations: recommendations.append("🔸 Continue current healthy lifestyle practices")
        contributions = get_risk_contributions(prediction_proba[1])
        
        st.session_state.prediction_data = {
            'risk_percentage': risk_percentage, 'recommendations': recommendations,
            'contributions': contributions, 'prediction': prediction
        }

# --- Results Display ---
if st.session_state.prediction_data:
    data = st.session_state.prediction_data
    risk_percentage = data['risk_percentage']
    recommendations = data['recommendations']
    contributions = data['contributions']
    prediction = data['prediction']
    
    st.markdown("---")
    
    # Risk Result Container
    risk_class = "high-risk" if prediction == 1 else "low-risk"
    alert_title = "⚠️ HIGH RISK DETECTED" if prediction == 1 else "✅ LOW RISK DETECTED"
    
    st.markdown(f"""
        <div class="result-container {risk_class}">
            <div class="risk-title">{alert_title}</div>
            <div class="risk-subtitle">Risk Probability: {risk_percentage:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_gauge, col_contribute = st.columns([1, 1.5])
    
    with col_gauge:
        # Risk Gauge 
        fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_percentage, title = {'text': "Risk Level (%)"}))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_contribute:
        # Risk Factor Contribution
        st.markdown("### 📈 Top Risk Factor Contribution (Simulated)")
        contributions_df = pd.DataFrame(list(contributions.items()), columns=['Factor', 'Impact (%)'])
        fig_bar = go.Figure(data=[go.Bar(x=contributions_df['Impact (%)'], y=contributions_df['Factor'], orientation='h', marker_color='#667eea')])
        st.plotly_chart(fig_bar, use_container_width=True)

    # What-If Analysis
    with st.expander("🔬 **What-If** Analysis: Lowering Your Risk", expanded=False):
        current_chol = st.session_state.user_inputs.get('chol', 200)
        target_chol = st.slider("Hypothetical Cholesterol (mg/dL)", 100, current_chol, max(100, current_chol - 40), key='whatif_chol')
        if target_chol < current_chol:
            simulated_drop = (current_chol - target_chol) * 0.005
            new_risk_prob = max(0.01, risk_percentage/100 - simulated_drop)
            st.info(f"Lowering Cholesterol could potentially drop your risk to **{new_risk_prob * 100:.1f}%**.")

    st.markdown("### 📋 Recommendations")
    for rec in recommendations: st.markdown(rec)
    
    st.download_button(
        label="⬇️ Download Personalized Health Report (HTML/PDF)",
        data=generate_report_html(st.session_state.user_inputs, risk_percentage, recommendations, contributions),
        file_name=f"CardioPredict_Report_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html", type="primary"
    )

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 CardioPredict | Powered by Machine Learning | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)