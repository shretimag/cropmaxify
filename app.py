import streamlit as st 
# import streamlit_tags as tags
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import pytorch
import matplot.lib
import seaborn
import sklearn



st.beta_set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def label(y,num):
    for i in range(y.shape[0]):
        a = y[i]
        return num[a]

def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAREDN;text-align:left;">CropMaxify Recommendation</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)



    with col2:
        st.subheader("Lets find out what crop would be best for you to grow in your farm!")
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
            x_np = np.array(N,P,K,temp, humidity, ph, rainfall).reshape((1,7))
            x_scaled = scaler.transform(x_np)
            xt = torch.from_numpy(x_scaled)
            output = (loaded_model(xt.float())).reshape((1,22))
            final = output.detach().numpy()
            Y_Pred = np.argmax(final,axis=1)
            ans = label(Y_Pred,num_crops)

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