import streamlit as st
import pickle
import numpy as np

st.title("Customer Personality Clustering App")

st.write("Enter customer details")

income = st.number_input("Annual Income")
spending = st.number_input("Spending Score")

model = pickle.load(open("mode.pkl","rb"))

if st.button("Predict Cluster"):

    data = np.array([[income, spending]])

    cluster = model.predict(data)

    st.success(f"Customer belongs to Cluster {cluster[0]}")
