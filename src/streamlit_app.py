import streamlit as st
import seaborn as sns
import pandas as pd
from knn_risk_classifier import assign_risk, train_knn

def main():
    st.title("Supplier Risk Classifier")

    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with supplier data", type=["csv"]
    )
    n_neighbors = st.sidebar.number_input(
        "KNN Neighbors", min_value=1, max_value=20, value=3, step=1
    )

    # --- FIX: WRAP POTENTIALLY FAILING CODE IN A TRY...EXCEPT BLOCK ---
    try:
        if uploaded_file is None:
            st.info("Upload a CSV file to run the classifier.")
            return

        df = pd.read_csv(uploaded_file)
        df = assign_risk(df)
        metrics, report, cm_df, model, scaler = train_knn(df, n_neighbors=int(n_neighbors))

        st.subheader("Metrics")
        st.json(metrics)

        st.subheader("Classification Report")
        st.dataframe(report)

        st.subheader("Pairplot")
        all_features = df.drop(columns=["Risk Classification"])
        preds = model.predict(scaler.transform(all_features))
        pairplot_df = all_features.copy()
        pairplot_df["Predicted Risk"] = preds
        
        g = sns.pairplot(pairplot_df, hue="Predicted Risk")
        st.pyplot(g.fig)

    except Exception as e:
        st.error(f"An error occurred during data processing or model training: {e}")
        st.warning("The script stopped before it could generate the plots. Please check the error message above and your terminal for a full traceback.")
        # Optionally, you can log the full traceback for more detailed debugging
        # import traceback
        # st.exception(traceback.format_exc())


if __name__ == "__main__":
    main()
