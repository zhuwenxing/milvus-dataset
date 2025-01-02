import streamlit as st

def dataset_page():
    if not st.session_state.current_storage:
        st.error("No storage selected")
        if st.button("Back to Storage"):
            st.session_state.page = 'storage'
            st.rerun()
        return

    st.title(f"Datasets in {st.session_state.current_storage}")

    # Back button
    if st.button("← Back to Storage"):
        st.session_state.page = 'storage'
        st.rerun()

    # Add new dataset section
    with st.expander("Add New Dataset", expanded=False):
        new_dataset_name = st.text_input("Dataset Name")
        dataset_description = st.text_area("Description")
        if st.button("Create Dataset"):
            if new_dataset_name:
                # TODO: Add dataset creation logic
                st.success(f"Dataset '{new_dataset_name}' created successfully!")
            else:
                st.error("Please fill in all required fields")

    # Display existing datasets
    st.subheader("Available Datasets")
    
    # Create a grid layout for dataset cards
    col1, col2, col3 = st.columns(3)
    
    # Example datasets (replace with actual dataset data)
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3"]
    
    for idx, dataset in enumerate(datasets):
        with [col1, col2, col3][idx % 3]:
            with st.container(border=True):
                st.write(f"### {dataset}")
                st.write("Records: 1000")
                st.write("Last modified: 2024-01-01")
                if st.button("View Details", key=f"view_{idx}"):
                    st.session_state.current_dataset = dataset
                    st.session_state.page = 'dataset_detail'
                    st.rerun()
