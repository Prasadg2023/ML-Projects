import streamlit as st

st.title("ML Project App")

st.write("Welcome to my deployed ML project 🚀")

# Example input
user_input = st.text_input("Enter something:")

if st.button("Submit"):
    st.write("You entered:", user_input)
