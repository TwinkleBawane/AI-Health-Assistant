import streamlit as st
from transformers import pipeline

# Load a medical Q&A model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

chatbot = load_model()

def healthcare_chatbot(question):
    context = (
        "Ibuprofen is a medicine used to reduce fever, pain, and inflammation. "
        "For nausea and vomiting, you can try drinking ginger tea, staying hydrated, and avoiding spicy food. "
        "Paracetamol is often recommended for fever."
    )
    
    response = chatbot(question=question, context=context)
    return response['answer']

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?")

    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant:", response)
        else:
            st.write("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
