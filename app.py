import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import PyPDF2

# Function to load Llama model and tokenizer
def load_llama_model():
    model_name = "llama-2-13b"  # or use smaller versions if necessary
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Function to extract text from a PDF
def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load Llama model
model, tokenizer = load_llama_model()

# Function to ask the Llama-based model questions using the extracted context
def ask_question_to_llm(question, context):
    prompt = f"Answer the following question using the context: {context}\n\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the response using Llama
    outputs = model.generate(**inputs, max_length=500)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app UI setup
def main():
    st.title("LLM-PDF AI Chat")
    
    # File upload functionality
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        context = extract_pdf_text(uploaded_file)
        st.text_area("Extracted Content", context, height=300)
        
        # Text input for the user's query
        question = st.text_input("Ask a question:")
        
        if question:
            # Get answer from Llama model
            answer = ask_question_to_llm(question, context)
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
