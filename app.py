import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import PyPDF2

# Function to load GPT-J model and tokenizer
def load_gptj_model():
    model_name = "EleutherAI/gpt-j-6B"  # GPT-J model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Function to extract text from a PDF
def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load GPT-J model
model, tokenizer = load_gptj_model()

# Function to ask GPT-J model questions using the extracted context
def ask_question_to_llm(question, context):
    prompt = f"Answer the following question using the context: {context}\n\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    # Generate the response using GPT-J
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app UI setup
def main():
    st.title("AI-LLM-financial-advisor-app")

    # File upload functionality
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        context = extract_pdf_text(uploaded_file)
        st.text_area("Extracted Content", context, height=300)
        
        # Text input for the user's query
        question = st.text_input("Ask a question:")

        if question:
            # Get answer from GPT-J model
            answer = ask_question_to_llm(question, context)
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
