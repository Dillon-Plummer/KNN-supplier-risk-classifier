import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Supplier Risk Classifier")

    # Initialize session state to manage the user's data source choice
    if 'data_source_choice' not in st.session_state:
        st.session_state.data_source_choice = None

    # --- 1. INITIAL CHOICE MODAL ---
    # This is the ONLY thing that runs if no choice has been made yet
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

    # --- 2. DATA SOURCE CONFIGURATION ---
    # This section runs only AFTER a choice has been made
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
                return # Stop if the file is invalid
        else:
            st.info("Please upload a file to continue.")
            return # Stop if no file has been uploaded yet

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

    # --- 3. MODEL CONFIGURATION ---
    n_neighbors = st.sidebar.number_input("KNN Neighbors", 1, 20, 3, 1)

    # --- 4. ANALYSIS AND VISUALIZATION ---
    if df is not None:
        try:
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Train model and get results
            metrics, report, cm_df, model, scaler = train_knn(df, n_neighbors=int(n_neighbors))

            st.subheader("Metrics")
            st.json(metrics)

            st.subheader("Classification Report")
            st.dataframe(report)

            # --- PAIRPLOT WITH PERFORMANCE FIXES ---
            st.subheader("Pairplot of Predicted vs. Actual Features")
            all_features = df.drop(columns=["Risk Classification"])
            preds = model.predict(scaler.transform(all_features))
            pairplot_df = all_features.copy()
            pairplot_df["Predicted Risk"] = preds
            
            st.info("Generating pairplot. This may take a moment for larger datasets.")

            # Show a spinner while the plot is being created
            with st.spinner("Building plot..."):
                SAMPLES_FOR_PLOT = 200  # Set a reasonable limit for plotting
                if len(pairplot_df) > SAMPLES_FOR_PLOT:
                    plot_df = pairplot_df.sample(SAMPLES_FOR_PLOT)
                else:
                    plot_df = pairplot_df

                # Generate the plot on the smaller, sampled dataframe
                g = sns.pairplot(plot_df, hue="Predicted Risk", diag_kind='hist')
                st.pyplot(g.fig)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
