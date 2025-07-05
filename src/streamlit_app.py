import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn

# --- Main App Logic ---
def main():
    st.title("Supplier Risk Classifier")

    # Initialize session state if it doesn't exist
    if 'data_source_choice' not in st.session_state:
        st.session_state.data_source_choice = None

    # --- MODAL: This is the ONLY thing that runs if no choice has been made ---
    if st.session_state.data_source_choice is None:
        with st.expander("Choose Your Data Source", expanded=True):
            st.write("Would you like to upload your own data or use the interactive demo data?")
            
            col1, col2 = st.columns(2)
            if col1.button("⬆️ Upload File", use_container_width=True):
                st.session_state.data_source_choice = 'upload'
                st.rerun()

            if col2.button("📊 Use Demo Data", use_container_width=True):
                st.session_state.data_source_choice = 'demo'
                st.rerun()
        
        # Stop execution completely until a button is clicked
        return

    # --- DATA SOURCE UI (This runs AFTER a choice is made) ---
    st.sidebar.header("Configuration")
    df = None

    if st.session_state.data_source_choice == 'upload':
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV or Excel file", type=["csv", "xlsx"]
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Please upload a file to continue.")
            return # Stop if no file is uploaded yet

    elif st.session_state.data_source_choice == 'demo':
        st.info("Using demo data, configured in the sidebar.")
        sc_increase = st.sidebar.slider("SC Increase Percentage", 1, 10, 4)
        qa_increase = st.sidebar.slider("QA Increase Percentage", 1, 10, 1)
        n_samples = st.sidebar.number_input("Number of Samples", 50, 1000, 350, 50)
        df = generate_dataset(
            sc_increase_percentage=sc_increase,
            qa_increase_percentage=qa_increase,
            n_samples=int(n_samples),
        )

    # --- MODEL CONFIGURATION & ANALYSIS ---
    n_neighbors = st.sidebar.number_input("KNN Neighbors", 1, 20, 3, 1)

    if df is not None:
        try:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            # (The rest of your analysis code for metrics and plots goes here)
            metrics, report, cm_df, model, scaler = train_knn(df, n_neighbors=int(n_neighbors))

            st.subheader("Metrics")
            st.json(metrics)

            st.subheader("Classification Report")
            st.dataframe(report)

            st.subheader("Pairplot of Predicted vs. Actual Features")
            all_features = df.drop(columns=["Risk Classification"])
            preds = model.predict(scaler.transform(all_features))
            pairplot_df = all_features.copy()
            pairplot_df["Predicted Risk"] = preds
            
            g = sns.pairplot(pairplot_df, hue="Predicted Risk")
            st.pyplot(g.fig)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
