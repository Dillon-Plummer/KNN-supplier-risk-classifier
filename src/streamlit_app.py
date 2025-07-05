import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn

# Initialize session state for the initial choice
if 'data_source_choice' not in st.session_state:
    st.session_state.data_source_choice = None

def main():
    st.title("Supplier Risk Classifier")

    # --- INITIAL CHOICE MODAL ---
    if st.session_state.data_source_choice is None:
        with st.expander("Choose your data source", expanded=True):
            st.write("Would you like to upload your own data or use the interactive demo data?")
            
            # Place buttons in columns for a clean layout
            col1, col2 = st.columns(2)
            if col1.button("⬆️ Upload File", key="upload", use_container_width=True):
                st.session_state.data_source_choice = 'upload'
                st.rerun()

            if col2.button("📊 Use Demo Data", key="demo", use_container_width=True):
                st.session_state.data_source_choice = 'demo'
                st.rerun()
        st.stop() # Stop the rest of the app from running until a choice is made

    # --- DATA LOADING AND CONFIGURATION ---
    df = None
    st.sidebar.header("Configuration")

    # Display UI based on the user's choice
    if st.session_state.data_source_choice == 'upload':
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            help="File must contain the required feature columns and 'Risk Classification'."
        )
        if uploaded_file:
            try:
                # Read the uploaded file into a dataframe
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")
                st.stop()
        else:
            st.info("Please upload a file to begin analysis.")
            st.stop()

    elif st.session_state.data_source_choice == 'demo':
        st.info("Using demo data. You can configure it in the sidebar.")
        sc_increase = st.sidebar.slider("SC Increase Percentage", 1, 10, 4)
        qa_increase = st.sidebar.slider("QA Increase Percentage", 1, 10, 1)
        n_samples = st.sidebar.number_input("Number of Samples", 50, 1000, 350, 50)
        
        df = generate_dataset(
            sc_increase_percentage=sc_increase,
            qa_increase_percentage=qa_increase,
            n_samples=int(n_samples),
        )

    # --- MODEL CONFIGURATION ---
    n_neighbors = st.sidebar.number_input("KNN Neighbors", 1, 20, 3, 1)

    # --- RUN ANALYSIS AND DISPLAY RESULTS ---
    if df is not None:
        try:
            st.subheader("Data Preview")
            st.dataframe(df.head())

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
            st.error(f"An error occurred during model training or plotting: {e}")
            st.warning("Ensure the data format is correct and contains the required columns.")

if __name__ == "__main__":
    main()
