import streamlit as st
import pandas as pd

def dataset_detail_page():
    if not st.session_state.current_dataset:
        st.error("No dataset selected")
        if st.button("Back to Datasets"):
            st.session_state.page = 'dataset'
            st.rerun()
        return

    st.title(f"Dataset: {st.session_state.current_dataset}")

    # Back button
    if st.button("← Back to Datasets"):
        st.session_state.page = 'dataset'
        st.rerun()

    # Dataset information
    with st.container(border=True):
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Total Records:** 1000")
            st.write("**Created Date:** 2024-01-01")
        with col2:
            st.write("**Last Modified:** 2024-01-01")
            st.write("**Storage:** " + st.session_state.current_storage)

    # Dataset viewer
    st.subheader("Data Preview")
    
    # Example data (replace with actual dataset data)
    example_data = pd.DataFrame({
        'id': range(1, 11),
        'value': [f"Value {i}" for i in range(1, 11)],
        'timestamp': pd.date_range(start='2024-01-01', periods=10)
    })
    
    st.dataframe(example_data, use_container_width=True)

    # README section
    st.subheader("README")
    with st.container(border=True):
        st.markdown("""
        ### Dataset Description
        This is an example dataset description.

        ### Schema
        - id (int): Primary key
        - value (string): Sample value
        - timestamp (datetime): Record timestamp

        ### Usage
        Example code for using this dataset:
        ```python
        # Load dataset
        data = load_dataset("example")
        ```
        """)
