import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🎓 Student Performance Prediction (Lasso Regression)")

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        st.write("🧾 Columns:", df.columns.tolist())

        # Select features & target
        st.subheader("⚙️ Select Features and Target")

        features = st.multiselect("Select Feature Columns", df.columns)
        target = st.selectbox("Select Target Column", df.columns)

        if features and target:

            X = df[features]
            y = df[target]

            # 🧠 HANDLE CATEGORICAL DATA

            # Convert categorical target
            if y.dtype == "object":
                st.warning("⚠️ Target column is categorical. Encoding applied.")
                y = pd.factorize(y)[0]

            # Convert categorical features
            X = pd.get_dummies(X, drop_first=True)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Alpha slider
            st.subheader("🎚️ Model Configuration")
            alpha = st.slider(
                "Select Alpha (Regularization Strength)",
                0.01, 1.0, 0.5
            )

            # Train Lasso model
            model = Lasso(alpha=alpha)
            model.fit(X_train_scaled, y_train)

            # Prediction
            y_pred = model.predict(X_test_scaled)

            # Evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📈 Model Performance")
            st.write(f"✅ Mean Squared Error: {mse:.2f}")
            st.write(f"✅ R² Score: {r2:.2f}")

            # Feature importance
            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            })

            st.subheader("📊 Feature Importance (Lasso)")
            st.write(coef_df)

            # Important features
            important = coef_df[coef_df["Coefficient"] != 0]

            st.subheader("🔍 Important Features")
            if not important.empty:
                st.write(important)
            else:
                st.warning("⚠️ No important features found. Try lowering alpha.")

        else:
            st.info("👆 Please select feature and target columns.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload a dataset to begin.")
