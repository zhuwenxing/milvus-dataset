import streamlit as st

def storage_page():
    st.title("Storage Management")
    
    # Add new storage section
    with st.expander("Add New Storage", expanded=False):
        new_storage_name = st.text_input("Storage Name")
        storage_path = st.text_input("Storage Path")
        if st.button("Create Storage"):
            if new_storage_name and storage_path:
                # TODO: Add storage creation logic
                st.success(f"Storage '{new_storage_name}' created successfully!")
            else:
                st.error("Please fill in all fields")

    # Display existing storages
    st.subheader("Available Storages")
    
    # Create a grid layout for storage cards
    col1, col2, col3 = st.columns(3)
    
    # Example storages (replace with actual storage data)
    storages = ["Storage 1", "Storage 2", "Storage 3"]
    
    for idx, storage in enumerate(storages):
        with [col1, col2, col3][idx % 3]:
            with st.container(border=True):
                st.write(f"### {storage}")
                st.write("Path: /path/to/storage")
                st.write("Datasets: 5")
                if st.button("Select", key=f"select_{idx}"):
                    st.session_state.current_storage = storage
                    st.session_state.page = 'dataset'
                    st.rerun()
