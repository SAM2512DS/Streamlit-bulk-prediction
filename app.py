from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import base64
from label import encoder

st.set_page_config(page_title='Mushrooms Prediction')
st.title('Mushroom Prediction 🍄')
st.subheader('Lets Find our Edible Mushrooms here !!! 🤤')

upload_file=st.file_uploader('Choose a csv')
if upload_file:
    st.markdown('-----')
    df=pd.read_csv(upload_file)
    st.dataframe(df)
    st.header("Your File Encoding is Done")
    data=encoder(df)
    encoder=LabelEncoder()
    for column in range(len(df.columns)):
       df[df.columns[column]]=encoder.fit_transform(df[df.columns[column]])
    st.dataframe(data)
    model = joblib.load("mushroomss.pkl")
    pred=pd.DataFrame(model.predict(df))
    pred.columns=['Results']
    result=pred.replace({1:'Edible' , 0:'Poisons'})
    st.dataframe(result)

download=st.button('Download prediction')
if download:
  'Download Started!'
  csv = result.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings
  linko= f'<a href="data:file/csv;base64,{b64}" download="mushrooms_prediction.csv">Download csv file</a>'
  st.markdown(linko, unsafe_allow_html=True)
  st.text('--------------------Created By : SAM A --------------------😊')


