import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.title('Hello World!')

model=joblib.load('knn_iris.pkl')
columns=joblib.load('columns.pkl')
scaler=joblib.load('scaler.pkl')

st.markdown("Provide Following Details")

sepal_length=st.number_input('Sepal Length',4.3,7.9,5.9)
sepal_width=st.number_input('Sepal Width',2.0,4.4,3.4)
petal_length=st.number_input('Petal Length',1.0,6.9,1.3)
petal_width=st.number_input('Petal Width',0.1,2.5,0.2)

if st.button('Predict'):
    data={'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}
    df=pd.DataFrame(data,index=[0])
    
    df=scaler.transform(df)
    result=model.predict(df)[0]
    st.write(result)