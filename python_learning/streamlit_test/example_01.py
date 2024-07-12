import streamlit as st
import pandas
import numpy

st.title("My first app")
st.write("here's our first attempt at using data to create a table:")
st.write(
    pandas.DataFrame({"first column": [1, 2, 3, 4], "second column": [1, 2, 3, 4]})
)

df = pandas.DataFrame({"first column": [1, 2, 3, 4], "second column": [1, 2, 3, 4]})
# df

map_data = pandas.DataFrame(
    numpy.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)

st.map(map_data)

if st.checkbox("Show dataframe"):
    chart_data=pandas.DataFrame(numpy.random.randn(20,3),columns=['a','b','c'])
    st.line_chart(chart_data)
    

option=st.sidebar.selectbox("which number do you like best?",df['first column'])    
'you selected: ',option