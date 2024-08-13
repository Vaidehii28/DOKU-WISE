import streamlit as st

if 'page' not in st.session_state:
    st.session_state.page = 'Document Level Extraction'  

st.sidebar.image('logo.png', width=275)

selection = st.sidebar.radio("Document upload options:", ["Document Level Extraction", "Aadhaar Masking"], index=["Document Level Extraction", "Aadhaar Masking"].index(st.session_state.page))


if selection != st.session_state.page:
    st.session_state.page = selection


if st.session_state.page == "Document Level Extraction":
    import tab1
    tab1.display()
elif st.session_state.page == "Aadhaar Masking":
    import tab2
    tab2.display()
