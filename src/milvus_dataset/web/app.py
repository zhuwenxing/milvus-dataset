import streamlit as st
from pages.storage_page import storage_page
from pages.dataset_page import dataset_page
from pages.dataset_detail_page import dataset_detail_page

st.set_page_config(
    page_title="Milvus Dataset Manager",
    layout="wide"
)

def main():
    # Initialize session state if not exists
    if 'current_storage' not in st.session_state:
        st.session_state.current_storage = None
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'page' not in st.session_state:
        st.session_state.page = 'storage'

    # Navigation logic
    if st.session_state.page == 'storage':
        storage_page()
    elif st.session_state.page == 'dataset':
        dataset_page()
    elif st.session_state.page == 'dataset_detail':
        dataset_detail_page()

if __name__ == "__main__":
    main()
