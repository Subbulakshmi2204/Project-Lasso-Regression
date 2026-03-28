import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🎓 Student Performance Prediction using Lasso Regression")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload dataset", 
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Check file type and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Features and target
    X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Internet_Usage']]
    y = df['Final_Score']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lasso model
    alpha = st.slider("Select Alpha (Regularization Strength)", 0.01, 1.0, 0.5)

    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Feature importance
    st.subheader("📊 Feature Importance (Lasso Coefficients)")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })

    st.write(coef_df)

    # Highlight important features
    st.subheader("🔍 Important Features")
    important = coef_df[coef_df['Coefficient'] != 0]
    st.write(important)

else:
    st.info("👆 Please upload your dataset to proceed.")
