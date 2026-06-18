import streamlit as st
from transformers import pipeline
import torch

# Page Configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Text Generation App")
st.write("Generate meaningful text using Hugging Face Pre-trained Foundation Models")

# Load model only once
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1  # CPU
    )
    return generator

generator = load_model()

# User Input
prompt = st.text_area(
    "Enter your prompt:",
    height=150,
    placeholder="Example: Artificial Intelligence is changing the world because..."
)

# Parameters
max_length = st.slider(
    "Maximum Length",
    min_value=50,
    max_value=300,
    value=100
)

temperature = st.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

# Generate Button
if st.button("🚀 Generate Text"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:

        with st.spinner("Generating..."):

            output = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )

            generated_text = output[0]["generated_text"]

        st.success("Text Generated Successfully!")

        st.subheader("Generated Text")

        st.write(generated_text)

        st.download_button(
            label="📥 Download Text",
            data=generated_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )