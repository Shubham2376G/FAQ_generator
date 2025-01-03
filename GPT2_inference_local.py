from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st


def generate_response(query):
    # Load fine-tuned model and tokenizer
    model_path = "gpt2"
    tokenizer1 = AutoTokenizer.from_pretrained(model_path)
    model1 = AutoModelForCausalLM.from_pretrained(model_path)
    # Generate responses
    chatbot = pipeline("text-generation", model=model1, tokenizer=tokenizer1)
    prompt = f"USER: {query} <|bot|>"
    response = chatbot(prompt, max_length=400, num_return_sequences=1)

    generated_text = response[0]['generated_text']
    response_text = generated_text.replace(prompt, '').strip()
    return response_text
    # return response[0]['generated_text']


def main():
    st.title("GPT2 Query-Response ")
    st.write("Enter your query below")

    # Input text box for query
    query = st.text_input("Your Query:", "")

    # Generate response when a query is entered
    if query:
        response = generate_response(query)
        st.write("**Response:**", response)

if __name__ == "__main__":
    main()
