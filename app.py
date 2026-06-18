import streamlit as st
from transformers import pipeline

# Load model
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

st.title("AI Text Generation App")
st.write("Generate meaningful text using Hugging Face models")

prompt = st.text_area(
    "Enter your prompt:",
    "Artificial Intelligence is"
)

if st.button("Generate Text"):

    result = generator(
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    st.subheader("Generated Text")

    st.write(result[0]["generated_text"])