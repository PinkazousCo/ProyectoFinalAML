#Aseguramos el control de rutas en python
import Definitions
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
import  streamlit_vertical_slider  as svs
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow
import keras
import sys
import numpy as np
#from io import StringIO

print("TensorFlow: ",tensorflow.version)
print("keras: ",keras.version)

from src.back.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Calidad Carnivora", page_icon="🥩"
)

col1, col2, col3,col4, col5, col6,col7, col8, col9,col10, col11, col12, = st.columns(12)

# st.set_page_config(layout="wide")
# st.subheader("Vertical Slider")
drop_fields = ["minute"]
target_feature = 'class'


### Support functions

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')
    

def highlight_diff(row):
    if row["Real"] != row["Predicción"]:
        # return ["background-color: blue"] * len(row)
        return ["background-color: #F5F5F5; color: black; font-weight: bold"] * len(row)
    return [""] * len(row)

def highlight_full_diff(row):
    if row["Predicción SVM"] != row["Predicción RF"]:
        # return ["background-color: blue"] * len(row)
        return ["background-color: #F5F5F5; color: black; font-weight: bold"] * len(row)
    return [""] * len(row)

def preprocess(df):
   df = df.drop_duplicates() # Eliminamos duplicados
   df = df.drop(drop_fields, axis = 1) # Eliminamos columnas que no vamos a usar
   class_labels = {'excellent': 1, 'good': 2, 'acceptable': 3, 'spoiled': 4}
   df['class'] = df['class'].replace(class_labels)
   X_data, y_variable = df.drop([target_feature], axis=1), df[target_feature]
   return X_data, y_variable


### My vars

ctrl = ModelController()

### My UI starting here

with col1:
    TVC=svs.vertical_slider("TVC",key="TVC",default_value=3551342097, step=0.01, min_value=1.232680, max_value=6.098218
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col2:
    MQ135=svs.vertical_slider("MQ135",key="MQ135",default_value=5.47, step=0.1, min_value=2.67, max_value=28.42
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col3:
    MQ136=svs.vertical_slider("MQ136",key="MQ136",default_value=6.29, step=0.1, min_value=0.22, max_value=86.39
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col4:
    MQ2=svs.vertical_slider("MQ2",key="MQ2",default_value=1.63, step=0.1, min_value=0.39, max_value=41.52
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col5:
    MQ3=svs.vertical_slider("MQ3",key="MQ3",default_value=6.61, step=0.1, min_value=3.26, max_value=26.75
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col6:
    MQ4=svs.vertical_slider("MQ4",key="MQ4",default_value=11.25, step=0.1, min_value=0.52, max_value=50
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col7:
    MQ5=svs.vertical_slider("MQ5",key="MQ5",default_value=2.82, step=0.01, min_value=0.91, max_value=79
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col8:
    MQ6=svs.vertical_slider("MQ6",key="MQ6",default_value=5.28, step=0.1, min_value=1.06, max_value=77.25
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col9:
    MQ8=svs.vertical_slider("MQ8",key="MQ8",default_value=3.81, step=0.1, min_value=0.79, max_value=54.27
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col10:
    MQ9=svs.vertical_slider("MQ9",key="MQ9",default_value=4.74, step=0.1, min_value=2.75, max_value=18.34
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col11:
    Humidity=svs.vertical_slider("Hum",key="Humidity",default_value=99.9, step=0.1, min_value=39.9, max_value=101
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
with col12:
    Temperature=svs.vertical_slider("Temp",key="Temperature",default_value=33.8, step=0.1, min_value=20, max_value=100
                        ,slider_color= 'green', track_color='lightgray',thumb_color = 'red'
                    )
minute=710
classe= "good"

data = {'minute': [minute], 'class': [classe], 'TVC': [TVC], 'MQ135': [MQ135], 'MQ136': [MQ136], 
        'MQ2': [MQ2], 'MQ3': [MQ3],'MQ4': [MQ4], 'MQ5': [MQ5],'MQ6': [MQ6], 'MQ8': [MQ8],
        'MQ9': [MQ9], 'Humidity': [Humidity],'Temperature': [Temperature]}
data = pd.DataFrame(data)

st.write(data)

colu1,colu2,colu3=st.columns(3)
target_feature = 'class'
X_test, Y_test = data.drop([target_feature], axis=1), data[target_feature]

#Ejecutamos modelos
result_svc=ctrl.predict_model_SVC(data)
result_rf=ctrl.predict_model_RF(data)
result_rn=ctrl.predict_model_RN(data)

with colu1:
    st.write("SVM Result",result_svc)

with colu2:
    st.write("RF Result",result_rf)

with colu3:
    st.write("NN Result",result_rn)


with st.form(key="my_form"):

    uploaded_file = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=False, type="csv"
    )

    submit_button = st.form_submit_button(label="Submit")

    with st.spinner("Processing your information...."):

        if submit_button and uploaded_file is not None:
            try:
                # Cargar la información del archivo csv
                bytes_data = uploaded_file.getvalue()
                st.write("Filename:", uploaded_file.name)

                #Asegurar la información de entrada como pandas dataframe
                input_df, is_valid = ctrl.load_input_data(bytes_data)

                if not is_valid:
                    st.warning("File structure not valid", icon="⚠️")

                # Presentamos la inforamción de forma tabulada o pestañas
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input Data", "NN", "SVM", "Random Forest", "Compare"])

                with tab1:
                    # Los datos deben ser presentados si son válidos, en este caso que pertenezcan a un pandas dataframe
                    if isinstance(input_df, pd.DataFrame) and not input_df.empty:
                        st.subheader("🥩 My Input Data")
                        st.dataframe(input_df)
                        resultado_svc=ctrl.predict_model_SVC(input_df)
                        resultado_rf=ctrl.predict_model_RF(input_df)
                        resultado_nn=ctrl.predict_model_RN(input_df)
                    else:
                        is_valid = False

                with tab2:
                    st.subheader("🧠 NN RESULTS")
                    nn_df=input_df
                    nn_df["NN"]=resultado_nn
                    st.dataframe(nn_df)
                with tab3:
                    st.subheader("🦾 SVM RESULTS")
                    svc_df=input_df
                    svc_df["SVM"]=resultado_svc
                    st.dataframe(svc_df)
                with tab4:
                    st.subheader("🌲 RANDOM FOREST RESULTS")
                    rf_df=input_df
                    rf_df["RF"]=resultado_rf
                    st.dataframe(rf_df)
                with tab5:
                    st.subheader("📈 TOTAL COMPARISON")
                    total_df=input_df
                    total_df["RF"]=resultado_rf
                    total_df["SVM"]=resultado_svc
                    total_df["NN"]=resultado_nn
                    total_df=total_df.drop(['TVC', 'MQ135', 'MQ136','MQ2', 'MQ3','MQ4','MQ5','MQ6','MQ8','MQ9','Humidity','Temperature'],axis="columns")
                    st.dataframe(total_df)
                if is_valid:
                    st.success("✅Done!")
            except:
                st.error("Something happened", icon="🚨")
        elif submit_button and uploaded_file is None:
            st.error("You must choose a csv file", icon="🚨")

