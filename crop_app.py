import joblib 
import streamlit as st

def set_bg_hack_url():    
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.pexels.com/photos/974314/pexels-photo-974314.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()


model=joblib.load(open('/content/crop_prediction.pkl','rb'))
scaler=joblib.load(open('/content/mscaler.pkl','rb'))


st.title('Crop Prediction')
N=st.number_input('ratio of nitrogen content in soil:')
P=st.number_input('ratio of phosphorus content in soil:')
K=st.number_input('ratio of pottasium content in soil:')
temperature=st.number_input('temperature in celcius:')
humidity=st.number_input('relative humidity in %:')
ph=st.number_input('ph of soil:')
rainfall=st.number_input('rainfall in mm:')
k=scaler.transform([[N,P,K,temperature,humidity,ph,rainfall]])
if st.button('Predict'):
  pred=model.predict(k)
  crop_prediction={20:'rice',11:'maize',3:'chickpea',9:'kidneybeans',
  18:'pigeonpeas',13:'mothbeans',14:'mungbean',2:'blackgram',10:'lentil'
  ,19:'pomegranate',1:'banana',12:'mango',7:'grapes',21:'watermelon',
  15:'muskmelon',0:'apple',16:'orange',17:'papaya',4:'coconut',
  6:'cotton',8:'jute',5:'coffee'}

  prediction_label=crop_prediction.get(pred[0],'unknown')

  st.markdown(
      f'<p style="font-size:20px; background-color:#333333; color:#ffffff; padding:10px;">'
      f'Predicted Crop : {prediction_label}'
      '</p>',
      unsafe_allow_html=True
  )

