import streamlit as st 
# import streamlit_tags as tags
import numpy as np
import os
import pickle
import warnings
import torch
import sklearn



st.beta_set_page_config(page_title="CropMaxify", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAREDN;text-align:left;">CropMaxify</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander("Information", expanded=True):
            st.write("""
            For helping farmers  in effectively utilise their land, they will be
suggested what crops they should grow on their land. They
will be asked to enter the region they live in, the season and,
the temperature of the surroundings. With the given details,
they will be suggested to grow the most suitable crop in the
given geographical conditions.
            """)



    with col2:
        st.subheader("Enter the following details. Nitrogen, Phophorous, Pottasium are the content of the nutrients in soils through fertilizers")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('ml.pkl')
            scaler = load_model('scaler.pkl')
            num_crops = load_model('num.pkl')
            x_np = (np.array([N,P,K,temp, humidity, ph, rainfall])).reshape((1,7))
            x_scaled = scaler.transform(x_np)
            xt = torch.from_numpy(x_scaled)
            output = (loaded_model(xt.float())).reshape((1,22))
            final = output.detach().numpy()
            Y_Pred = np.argmax(final,axis=1)	
            ans = num_crops[Y_Pred]

            col1.write('''
		    ## Results 
		    ''')
            col1.success(f"{ans} are recommended by us for your farm")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
