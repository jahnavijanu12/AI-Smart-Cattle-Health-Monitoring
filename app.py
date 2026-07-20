import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cattle Health Monitoring System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #28a745;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
    .risk-medium {
        background-color: #ffc107;
        padding: 0.5rem;
        border-radius: 5px;
        color: black;
    }
    .risk-high {
        background-color: #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data(ttl=3600)
def generate_synthetic_data(n_samples=500):
    """Generate more realistic synthetic cattle health data"""
    np.random.seed(42)
    
    # Generate physiological parameters with realistic correlations
    data = pd.DataFrame()
    
    # Body Temperature (normal: 38-39.5°C)
    data["Body Temperature"] = np.random.normal(38.7, 0.5, n_samples)
    data.loc[data["Body Temperature"] > 40.5, "Body Temperature"] = 40.5
    data.loc[data["Body Temperature"] < 37.5, "Body Temperature"] = 37.5
    
    # Heart Rate (normal: 60-80 bpm)
    data["Heart Rate"] = np.random.normal(70, 8, n_samples)
    data.loc[data["Heart Rate"] > 100, "Heart Rate"] = 100
    data.loc[data["Heart Rate"] < 50, "Heart Rate"] = 50
    
    # Respiration Rate (normal: 20-35 breaths/min)
    data["Respiration Rate"] = np.random.normal(28, 5, n_samples)
    data.loc[data["Respiration Rate"] > 50, "Respiration Rate"] = 50
    data.loc[data["Respiration Rate"] < 15, "Respiration Rate"] = 15
    
    # Rumination Time (normal: 450-550 minutes/day)
    data["Rumination Time"] = np.random.normal(500, 50, n_samples)
    data.loc[data["Rumination Time"] > 650, "Rumination Time"] = 650
    data.loc[data["Rumination Time"] < 350, "Rumination Time"] = 350
    
    # Activity Level (normal: 250-350)
    data["Activity Level"] = np.random.normal(300, 40, n_samples)
    data.loc[data["Activity Level"] > 450, "Activity Level"] = 450
    data.loc[data["Activity Level"] < 150, "Activity Level"] = 150
    
    # Weight (kg) - normal: 400-500 kg
    data["Weight"] = np.random.normal(450, 35, n_samples)
    data.loc[data["Weight"] > 600, "Weight"] = 600
    data.loc[data["Weight"] < 350, "Weight"] = 350
    
    # Age (years)
    data["Age"] = np.random.randint(2, 12, n_samples)
    
    # Health Status Labeling with weighted logic
    labels = []
    for i in range(len(data)):
        score = 0
        
        # Temperature abnormalities
        if data["Body Temperature"][i] > 39.5:
            score += 40
        elif data["Body Temperature"][i] < 38.0:
            score += 20
            
        # Rumination abnormalities
        if data["Rumination Time"][i] < 430:
            score += 30
        elif data["Rumination Time"][i] < 460:
            score += 10
            
        # Activity abnormalities
        if data["Activity Level"][i] < 240:
            score += 20
        elif data["Activity Level"][i] < 270:
            score += 10
            
        # Heart rate abnormalities
        if data["Heart Rate"][i] > 85:
            score += 15
        elif data["Heart Rate"][i] > 95:
            score += 25
            
        # Respiration abnormalities
        if data["Respiration Rate"][i] > 35:
            score += 10
            
        # Age factor (older cattle more susceptible)
        if data["Age"][i] > 8:
            score += 5
            
        # Classify based on score
        if score >= 60:
            labels.append(2)  # Diseased
        elif score >= 30:
            labels.append(1)  # At Risk
        else:
            labels.append(0)  # Healthy
    
    data["Health Status"] = labels
    return data

@st.cache_resource
def train_model(data):
    """Train and evaluate the machine learning model"""
    X = data.drop("Health Status", axis=1)
    y = data["Health Status"]
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with hyperparameter optimization
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    return model, scaler, X_train, X_test, y_train, y_test, accuracy, cv_scores

def calculate_health_risk_score(parameters):
    """Calculate comprehensive health risk score"""
    body_temp, heart_rate, resp_rate, rumination, activity, weight, age = parameters
    
    # Normalize parameters (0-1 scale, where 1 indicates highest risk)
    temp_risk = max(0, min(1, abs(body_temp - 38.7) / 2.0))
    heart_risk = max(0, min(1, abs(heart_rate - 70) / 30))
    resp_risk = max(0, min(1, abs(resp_rate - 28) / 20))
    rum_risk = max(0, min(1, (500 - rumination) / 200))
    activity_risk = max(0, min(1, (300 - activity) / 200))
    weight_risk = max(0, min(1, abs(weight - 450) / 150))
    age_risk = max(0, min(1, (age - 2) / 13))
    
    # Weighted risk score
    risk_score = (
        0.25 * temp_risk +
        0.15 * heart_risk +
        0.10 * resp_risk +
        0.20 * rum_risk +
        0.15 * activity_risk +
        0.05 * weight_risk +
        0.10 * age_risk
    )
    
    return min(1.0, risk_score)

def display_risk_indicator(risk_score):
    """Display risk indicator with color coding"""
    if risk_score < 0.3:
        st.markdown('<div class="risk-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
        return "Low Risk", "green"
    elif risk_score < 0.6:
        st.markdown('<div class="risk-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
        return "Medium Risk", "orange"
    else:
        st.markdown('<div class="risk-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
        return "High Risk", "red"

# Main Application
def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🐄 Advanced Cattle Health Monitoring System")
    st.markdown("### AI-Powered Health Prediction with Explainable AI")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate or load data
    with st.spinner("Loading data and training model..."):
        data = generate_synthetic_data(500)
        model, scaler, X_train, X_test, y_train, y_test, accuracy, cv_scores = train_model(data)
    
    # Display model performance metrics
    with st.expander("📊 Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("CV Score (Avg)", f"{cv_scores.mean():.2%}")
        with col3:
            st.metric("CV Score (Std)", f"{cv_scores.std():.2%}")
    
    # Sidebar inputs
    st.sidebar.header("🐄 Animal Health Parameters")
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        body_temp = st.number_input("🌡️ Body Temperature (°C)", 37.0, 42.0, 38.7, step=0.1)
        heart_rate = st.number_input("💓 Heart Rate (bpm)", 40, 120, 70, step=1)
        resp_rate = st.number_input("🌬️ Respiration Rate", 10, 60, 28, step=1)
        rumination = st.number_input("🔄 Rumination Time", 300, 700, 500, step=10)
    
    with col2:
        activity = st.number_input("🏃 Activity Level", 150, 500, 300, step=10)
        weight = st.number_input("⚖️ Weight (kg)", 350, 650, 450, step=5)
        age = st.number_input("📅 Age (years)", 1, 15, 5, step=1)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        threshold_adjust = st.slider("Risk Threshold Adjustment", 0.0, 1.0, 0.5, 0.05)
    
    # Prediction button
    if st.sidebar.button("🔍 Analyze Health Status", type="primary", use_container_width=True):
        
        # Prepare input data
        input_data = np.array([[body_temp, heart_rate, resp_rate,
                                rumination, activity, weight, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Calculate risk score
        risk_score = calculate_health_risk_score([body_temp, heart_rate, resp_rate,
                                                  rumination, activity, weight, age])
        
        # Status mapping
        status_map = {0: "Healthy", 1: "At Risk", 2: "Diseased"}
        status_colors = {0: "green", 1: "orange", 2: "red"}
        
        # Main results section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("📊 Diagnosis Result")
            st.markdown(f"### Health Status: **:{status_colors[prediction]}[{status_map[prediction]}]**")
            
            # Risk indicator
            risk_level, risk_color = display_risk_indicator(risk_score)
            st.metric("Health Risk Score", f"{risk_score:.2%}", delta=None)
            
        with col2:
            st.subheader("🎯 Prediction Confidence")
            for i, status in enumerate(["Healthy", "At Risk", "Diseased"]):
                st.progress(probabilities[i], text=f"{status}: {probabilities[i]:.1%}")
                
        with col3:
            st.subheader("📈 Parameter Status")
            param_status = []
            if body_temp > 39.5:
                param_status.append("⚠️ Elevated Temperature")
            if rumination < 430:
                param_status.append("⚠️ Low Rumination")
            if activity < 240:
                param_status.append("⚠️ Low Activity")
            if heart_rate > 85:
                param_status.append("⚠️ Elevated Heart Rate")
            
            if param_status:
                for status in param_status:
                    st.warning(status)
            else:
                st.success("All parameters within normal range")
        
        # Feature importance visualization
        st.markdown("---")
        st.subheader("📊 Feature Impact Analysis")
        
        # Create feature importance plot
        feature_importance = pd.DataFrame({
            'feature': data.drop('Health Status', axis=1).columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                     orientation='h', title='Feature Importance',
                     color='importance', color_continuous_scale='Viridis')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 0 and risk_score < 0.3:
            st.success("✅ Animal appears healthy. Continue regular monitoring.")
        elif prediction == 1 or risk_score >= 0.3:
            st.warning("⚠️ **Action Required:** Monitor closely. Consider consulting veterinarian.")
            recommendations = []
            if body_temp > 39.5:
                recommendations.append("• Check for signs of fever or infection")
            if rumination < 430:
                recommendations.append("• Monitor feed intake and digestive health")
            if activity < 240:
                recommendations.append("• Encourage movement and check for lameness")
            if heart_rate > 85:
                recommendations.append("• Check for signs of stress or cardiac issues")
            
            for rec in recommendations:
                st.write(rec)
        else:
            st.error("🚨 **URGENT:** Immediate veterinary attention required!")
            st.write("• Isolate animal if necessary")
            st.write("• Contact veterinarian immediately")
            st.write("• Monitor vital signs closely")
        
        # Download report option
        st.markdown("---")
        if st.button("📄 Download Health Report"):
            report_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Health Status": status_map[prediction],
                "Risk Score": f"{risk_score:.2%}",
                "Confidence": f"{probabilities[prediction]:.1%}",
                **{f"Parameter_{k}": v for k, v in zip(data.columns[:-1], input_data[0])}
            }
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"cattle_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Additional information in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About This System"):
        st.markdown("""
        **Features:**
        - Random Forest Classifier
        - SHAP & LIME explanations
        - Real-time risk assessment
        - Comprehensive recommendations
        
        **Parameters Monitored:**
        - Body Temperature
        - Heart Rate
        - Respiration Rate
        - Rumination Time
        - Activity Level
        - Weight
        - Age
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        🐄 Advanced Cattle Health Monitoring System | Powered by Machine Learning & Explainable AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()