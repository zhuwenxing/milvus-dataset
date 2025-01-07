from datetime import datetime

import streamlit as st


class DatasetViewer:
    def __init__(self):
        if "page" not in st.session_state:
            st.session_state.page = "config"
        if "storage_config" not in st.session_state:
            st.session_state.storage_config = {}
        if "datasets" not in st.session_state:
            st.session_state.datasets = []

    def config_page(self):
        st.title("Storage Configuration")

        storage_type = st.selectbox("Storage Type", ["Local", "S3", "Azure Blob"])

        if storage_type == "Local":
            path = st.text_input("Local Path")
            if st.button("Save"):
                st.session_state.storage_config = {"type": "local", "path": path}
                st.session_state.page = "list"
                st.rerun()

        elif storage_type == "S3":
            bucket = st.text_input("Bucket Name")
            access_key = st.text_input("Access Key")
            secret_key = st.text_input("Secret Key", type="password")
            if st.button("Save"):
                st.session_state.storage_config = {
                    "type": "s3",
                    "bucket": bucket,
                    "access_key": access_key,
                    "secret_key": secret_key,
                }
                st.session_state.page = "list"
                st.rerun()

        elif storage_type == "Azure Blob":
            container = st.text_input("Container Name")
            connection_string = st.text_input("Connection String", type="password")
            if st.button("Save"):
                st.session_state.storage_config = {
                    "type": "azure",
                    "container": container,
                    "connection_string": connection_string,
                }
                st.session_state.page = "list"
                st.rerun()

    def list_page(self):
        st.title("Datasets")

        if not st.session_state.datasets:
            st.session_state.datasets = [
                {
                    "name": "dataset1",
                    "description": "Example dataset 1",
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "size": "1.2GB",
                    "samples": 10000,
                },
                {
                    "name": "dataset2",
                    "description": "Example dataset 2",
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "size": "2.5GB",
                    "samples": 20000,
                },
            ]

        search = st.text_input("Search datasets")

        for dataset in st.session_state.datasets:
            if search.lower() in dataset["name"].lower():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.subheader(dataset["name"])
                        st.write(dataset["description"])
                    with col2:
                        st.write(f"Created: {dataset['created_at']}")
                        st.write(f"Size: {dataset['size']}")
                    with col3:
                        if st.button("View Details", key=dataset["name"]):
                            st.session_state.current_dataset = dataset
                            st.session_state.page = "detail"
                            st.rerun()
                st.divider()

        if st.button("Change Storage Config"):
            st.session_state.page = "config"
            st.rerun()

    def detail_page(self):
        st.title(f"Dataset: {st.session_state.current_dataset['name']}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Details")
            st.write(f"Description: {st.session_state.current_dataset['description']}")
            st.write(f"Created: {st.session_state.current_dataset['created_at']}")
            st.write(f"Size: {st.session_state.current_dataset['size']}")
            st.write(f"Number of samples: {st.session_state.current_dataset['samples']}")

        with col2:
            st.write("### Storage Info")
            st.json(st.session_state.storage_config)

        st.write("### Data Preview")
        st.dataframe({"column1": [1, 2, 3], "column2": ["a", "b", "c"], "column3": [1.1, 2.2, 3.3]})

        if st.button("Back to List"):
            st.session_state.page = "list"
            st.rerun()

    def run(self):
        if st.session_state.page == "config":
            self.config_page()
        elif st.session_state.page == "list":
            self.list_page()
        elif st.session_state.page == "detail":
            self.detail_page()


if __name__ == "__main__":
    app = DatasetViewer()
    app.run()
