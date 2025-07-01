import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, assign_risk, train_knn


def main():
    st.title("Supplier Risk Classifier")

    st.sidebar.header("Configuration")
    sc_increase = st.sidebar.slider(
        "SC Increase Percentage", min_value=1, max_value=10, value=4
    )
    qa_increase = st.sidebar.slider(
        "QA Increase Percentage", min_value=1, max_value=10, value=1
    )
    n_samples = st.sidebar.number_input(
        "Number of Samples", min_value=50, max_value=1000, value=350, step=50
    )
    n_neighbors = st.sidebar.number_input(
        "KNN Neighbors", min_value=1, max_value=20, value=3, step=1
    )

    df = generate_dataset(
        sc_increase_percentage=sc_increase,
        qa_increase_percentage=qa_increase,
        n_samples=int(n_samples),
    )
    df = assign_risk(df)
    metrics, report, cm_df = train_knn(df, n_neighbors=int(n_neighbors))

    st.subheader("Metrics")
    st.json(metrics)

    st.subheader("Classification Report")
    st.dataframe(report)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
