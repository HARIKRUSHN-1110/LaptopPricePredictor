import streamlit as st
import pickle
import numpy as np
st.title('LAPTOP PRICE ESTIMATOR')
pipe=pickle.load(open('pipelap.pkl','rb'))
data=pickle.load(open('lapdata.pkl','rb'))
company=st.selectbox('brand',data['Company'].unique())
type=st.selectbox('type',data['TypeName'].unique())
ram=st.selectbox('RAM(in GigaByte)',data['Ram'].unique())
weight=st.number_input('Laptop Weight')
touch=st.selectbox('Touchscreen',['YES','NO'])
IPS=st.selectbox('IPS',['YES','NO'])
screen_size=st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU',data['processor_brand'].unique())
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU',data['Gpu_brand'].unique())
os = st.selectbox('OS',data['OS'].unique())
if st.button('Predict'):
    ppi=None
    if touch=='YES':
        touch=1
    else:
        touch=0
    if IPS=='YES':
        IPS=1
    else:
        IPS=0
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query=np.array([company,type,ram,weight,touch,IPS,ppi,cpu,hdd,ssd,gpu,os],dtype=object)
    query=query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
