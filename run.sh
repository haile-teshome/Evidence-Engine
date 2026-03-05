#!/bin/bash
# Start Streamlit with file watcher disabled to avoid torch compatibility issues
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
streamlit run app.py
