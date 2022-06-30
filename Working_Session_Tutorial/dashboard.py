import streamlit as st
import pickle as pkl
import pandas as pd
import xgboost
import numpy as np

#read in dataset
df=pd.read_csv('Data/heart.csv')

#load the model into the dash
f=open('Models/model.pkl','rb')
model=pkl.load(f)
f.close()

#Create the header
st.markdown("<h1 style='text-align: center; color: "+'black'+";'>"+'Heart Disease Dashboard'+"</h1>", unsafe_allow_html=True)

#Instantiate the columns
#This will divide our dash into three portions
col1,col2,col3=st.columns(3)
#Let's make a widget
age=st.slider('Age',min(df['age']),max(df['age']))
#Widgets in our first column
with col1:
    #Let's create a variety of widgets for each input
    restecg=st.selectbox('Resting ECG',list(set(df['restecg'])))
    sex=st.selectbox('Gender',list(set(df['sex'])))
    cp=st.radio('Chest Pain Type 0-4',list(set(df['cp'])))
    thalach=st.number_input('Maximum Heart Rate',min(df['thalach']),max(df['thalach']))
    slope=st.selectbox('Slope at Peak Exercise ST Segment',list(set(df['slope'])))
    fbs=st.checkbox('Fasting Blood Sugar > 120 mg/dl',list(set(df['fbs'])))
#Widgets in our second column
with col2:
    #Let's create a column for space
    st.write(' ')
#Widgets in our thrid column
with col3:
    #Let's create a variety of widgets for each input
    rest_bps=st.number_input('Resting Beats per Second',min(df['trestbps']),max(df['trestbps']))
    exang=st.checkbox('Exercise Induced Angina',list(set(df['exang'])))
    oldpeak=st.slider('ST Depression Induced by Exercise Relative to Rest',min(df['oldpeak']),max(df['oldpeak']))
    thal=st.radio('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',list(set(df['thal'])))
    chol=st.slider('Serum Cholesteral in mg/dl',min(df['chol']),max(df['chol']))
    ca=st.selectbox('Number of Major Vessels (0-3) Colored by Flourosopy',list(set(df['ca'])))


#input=dict(zip(list(df.columns),[age,sex,cp,rest_bps,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]))
#put all of our values into an input
df_input=np.array([age,sex,cp,rest_bps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1)
prediction=list(model.predict(df_input))[0]
#Create an output dictionary
pred_dict={0:'No Disease',1:'Heart Disease'}
#Create a color dictionary
pred_color={0:'blue',1:'red'}
#Create output
st.markdown("<h1 style='text-align: center; color: "+pred_color[prediction]+";'>"+pred_dict[prediction]+"</h1>", unsafe_allow_html=True)
