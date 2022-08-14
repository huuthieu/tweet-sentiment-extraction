import streamlit as st
from model import predict, initialize


models, tokenizers = initialize()

st.set_page_config(page_title="TSE", page_icon="🤖")
# Header
st.title("Tweet Sentiment Extraction")

# Form to add your items
with st.form("my_form"):
    # Textarea to type the source text.
    user_input = st.text_area("Tweet text", max_chars=200)
    option = st.selectbox(
    "Select Sentiment",
    ("Positive", "Negative", "Neutral"))
    # Translate with CTranslate2 model
    # translation = translate(user_input, translator, sp_source_model, sp_target_model)
    output = user_input.upper()

    # Create a button
    submitted = st.form_submit_button("Extract")
    # If the button pressed, print the translation
    # Here, we use "st.info", but you can try "st.write", "st.code", or "st.success".
    if submitted:
        with st.spinner('Wait for it...'):
            selected_text = predict(user_input,option.lower(), models, tokenizers)
        # st.write(selected_text)
        st.info(selected_text)