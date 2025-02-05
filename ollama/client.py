import requests
import streamlit as st

# Function to send input to FastAPI and fetch response
def get_response_from_api(question):
    try:
        response = requests.post(
            "http://localhost:8000/invoke",  # FastAPI endpoint
            json={
                "input": {"question": question},
                "config": {},
                "kwargs": {}
            }
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json().get('output', 'No output in response')  # Safely access 'output'

    except requests.exceptions.RequestException as e:
        return f"Error communicating with the API: {e}"
    except KeyError:
        return "Unexpected response format from API"

# Streamlit interface
st.title("LangChain Demo with LLAMA3.2")
input_text = st.text_input("Enter your question or topic:")

if input_text:
    response = get_response_from_api(input_text)
    st.write(response)
