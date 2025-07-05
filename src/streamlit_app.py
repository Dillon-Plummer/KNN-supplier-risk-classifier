import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn, preprocess_data

# --- Constants ---
TARGET_COLUMN = "Risk Classification"

# --- Main App Logic ---
def main():
    st.title("Supplier Risk Classifier")

    if 'data_source_choice' not in st.session_state:
        st.session_state.data_source_choice = None

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
        return

    st.sidebar.header("Configuration")
    df = None

    if st.session_state.data_source_choice == 'upload':
        uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Please upload a file to continue.")
            return

    elif st.session_state.data_source_choice == 'demo':
        st.info("Using demo data, configured in the sidebar.")
        sc_increase = st.sidebar.slider("SC Increase Percentage", 1, 10, 4)
        qa_increase = st.sidebar.slider("QA Increase Percentage", 1, 10, 1)
        n_samples = st.sidebar.number_input("Number of Samples", 50, 1000, 350, 50)
        df = generate_dataset(sc_increase_percentage=sc_increase, qa_increase_percentage=qa_increase, n_samples=int(n_samples))

    n_neighbors = st.sidebar.number_input("KNN Neighbors", 1, 20, 3, 1)

    if df is not None:
        try:
            processed_df, dropped_rows, categorical_cols = preprocess_data(df)
            
            if dropped_rows > 0:
                st.warning(f"Warning: {dropped_rows} rows were dropped due to missing data.")
            
            if categorical_cols:
                st.info(f"The following categorical columns were identified and encoded: {', '.join(categorical_cols)}")

            st.subheader("Data Preview (After Processing)")
            st.dataframe(processed_df.head())

            metrics, report, cm_df, model, scaler = train_knn(processed_df, n_neighbors=int(n_neighbors))

            st.subheader("Metrics")
            st.json(metrics)
            st.subheader("Classification Report")
            st.dataframe(report)

            st.subheader("Pairplot of Predicted vs. Actual Features")
            all_features = processed_df.drop(columns=[TARGET_COLUMN])
            preds = model.predict(scaler.transform(all_features))
            pairplot_df = all_features.copy()
            pairplot_df["Predicted Risk"] = preds
            
            st.info("Generating pairplot from a sample of the data. This may take a moment.")
            with st.spinner("Building plot..."):
                SAMPLES_FOR_PLOT = 200
                plot_df = pairplot_df.sample(min(len(pairplot_df), SAMPLES_FOR_PLOT))
                g = sns.pairplot(plot_df, hue="Predicted Risk", diag_kind='hist')
                st.pyplot(g.fig)

        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
