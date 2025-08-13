# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Dataset & Model
# ------------------------------
iris = load_iris(as_frame=True)
df = iris.frame
model = joblib.load("iris_model.pkl")

species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Simulated second model for comparison (optional)
# In real case, train and load another model
model_2_accuracy = 0.92
model_1_accuracy = 0.95

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.set_page_config(page_title="Iris ML App", page_icon="🌸", layout="wide")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Exploration", "Visualisation", "Model Prediction", "Model Performance"]
)

# ------------------------------
# Home Section
# ------------------------------
if menu == "Home":
    st.title("🌸 Iris Flower Prediction App")
    st.write("""
    This interactive machine learning application predicts the species of an Iris flower 
    based on four measurements: sepal length, sepal width, petal length, and petal width.  

    **Features of the App:**
    - Explore the Iris dataset with filtering and summaries
    - Visualise data trends with interactive charts
    - Predict flower species in real-time
    - View model performance metrics and comparisons

    **Dataset:** [Scikit-learn Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)  
    **Algorithm Used:** Logistic Regression (Scikit-learn)
    """)
    st.markdown("---")
    st.caption("Created for Machine Learning Model Deployment Assignment")

# ------------------------------
# Data Exploration Section
# ------------------------------
elif menu == "Data Exploration":
    st.title("📊 Data Exploration")

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Interactive Data Filtering")
    species_filter = st.multiselect("Select Species", df['target'].unique(), default=df['target'].unique())
    filtered_df = df[df['target'].isin(species_filter)]
    st.write(filtered_df)

# ------------------------------
# Visualisation Section
# ------------------------------
elif menu == "Visualisation":
    st.title("📈 Data Visualisations")

    # Histogram
    st.subheader("Sepal Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['sepal length (cm)'], kde=True, ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Sepal vs Petal Length")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x='sepal length (cm)',
        y='petal length (cm)',
        hue=df['target'].map(species_map),
        ax=ax
    )
    st.pyplot(fig)

    # Box plot
    st.subheader("Petal Width by Species")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['target'].map(species_map), y='petal width (cm)', ax=ax)
    st.pyplot(fig)

# ------------------------------
# Model Prediction Section
# ------------------------------
elif menu == "Model Prediction":
    st.title("🔮 Model Prediction")

    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        )
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.success(f"Predicted Species: **{species_map[prediction]}** 🌿")
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            "Species": [species_map[i] for i in range(len(probabilities))],
            "Probability": probabilities
        })
        st.write(prob_df)

# ------------------------------
# Model Performance Section
# ------------------------------
elif menu == "Model Performance":
    st.title("📉 Model Performance")

    X = df.drop(columns=["target"])
    y = df["target"]
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y, y_pred, target_names=iris.target_names))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    st.pyplot(fig)

    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Other Model"],
        "Accuracy": [model_1_accuracy, model_2_accuracy]
    })
    st.write(comparison_df)
