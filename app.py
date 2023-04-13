# ========== IMPORT LIBRARIES ========== #
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sw
import streamlit.components.v1 as components
import codecs 
import sklearn
from streamlit_option_menu import option_menu
from scipy import stats
import datetime
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import PredictionError
# ========== END IMPORT LIBRARIES ========== #

# ========== SETTINGS MENU ========== #
date = datetime.today().year
st.set_page_config(page_title="house prediction - app", page_icon="📍") # , layout="wide"
with st.sidebar:
    
    selected = option_menu(
        "MENU",
        ["HOME", "DATASET", "EDA", "VISUALIZATION", "PREDICTION"],
        icons=["house", "graph-up-arrow", "book", "bar-chart-line", "calendar2-check"],
        menu_icon = "cast",
        default_index=0
    )

html_temp = """
    <div style="background-color:#d33682;padding:15px;text-align:center;">
    <h2 style="color:white;">House Prediction App</h2>
    </div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
        
hide_menu = """
    <style>
        #MainMenu{
            visibility: hidden;
        }
        
        
        footer{
            visibility: hidden;
         }
         
         header{
             visibility: visible;
         }
        
        {
            display:block;
            position:relative;
            color:tomato;
            padding:5px;
            top:3px;
            background-color:tomato;
            text-align:center;
            content:'Copyright @ 2022: Davila Rostaing';
        } 
    </style>
"""
# ========== END SETTINGS MENU ========== #

# ========== LOAD DATA ========== #
@st.cache_data
def load_data():
    dt = pd.read_csv("dataset/ParisHousing.csv")
    df = dt.copy()
    df["hasPool"].replace([0, 1], ["No", "Yes"], inplace=True)
    df["hasYard"].replace([0, 1], ["No", "Yes"], inplace=True)
    df["isNewBuilt"].replace([0, 1], ["No", "Yes"], inplace=True)
    df["hasStormProtector"].replace([0, 1], ["No", "Yes"], inplace=True)
    df["hasStorageRoom"].replace([0, 1], ["No", "Yes"], inplace=True)
    return df
df = load_data()
# ========== END LOAD DATA ========== #

# ========== EDA ========== #
def eda():
    
    st.write("***Exploratory Data Analysis (EDA)***")
    sb = st.sidebar.selectbox("Exploratory Data Analysis (EDA)", ["shape", "types", "dimension", "size", "significant_correlation", "describe", "null values", "Hypothesis"])
    
    if sb == "shape":
        st.write(f"{sb}:", df.shape)
        
    elif sb == "types":
        st.write(df.dtypes.to_frame("Types"))
        
    elif sb == "dimension":
        st.write(f"{sb}:", df.ndim)
        
    elif sb == "size":
        st.write(f"{sb}:", df.size)
        
    elif sb == "significant_correlation":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        pearson_coef, p_value = stats.pearsonr(df[var_x], df[var_y])
        st.write(f"The Pearson Correlation Coefficient is {pearson_coef}, with P-value = {p_value}")
        if p_value < 0.001:
            st.success("There is strong evidence that the correlation is significant.")
            st.balloons()
        elif p_value < 0.05:
            st.info("There is moderate evidence that the correlation is significant.")
        elif p_value < 0.1:
            st.warning("There is weak evidence that the correlation is significant.")
        elif p_value >  0.1:
            st.error("There is no evidence that the correlation is significant.")
        
    elif sb == "describe":
        st.write(df.describe())
        
    elif sb == "null values":
        st.write(df.isnull().sum().to_frame("Null values"))
        
    elif sb == "Hypothesis":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        tset, p_value = stats.pearsonr(df[var_x], df[var_y])
        st.write(f"Statistique = {tset} P-value = {p_value}")
        if p_value < 0.05:
            st.warning("We are rejecting null hypothesis.")
        else:
            st.success("We are accepting null hypothesis.")
# ========== END EDA ========== # 

# ========== VISUALIZATION ========== # 
def st_display_seetviz(report_html, width=1000, height=500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)   
           
def visual():
    
    choose_viz = st.sidebar.selectbox("Visualization (Charts)", ("Bar", "SweetViz", "Correlation", "Line", "Histogram", "Violin", "Density", "Heatmaps", "Scatter", "Area", "Funnel", "Pie", "Box", "Ecdf"))
     
    if choose_viz == "Bar":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.bar(df, x=var_x, y=var_y, color=var_color)
        st.plotly_chart(fig)
    
    elif choose_viz == "SweetViz":
        report = sw.analyze(df)
        report.show_html()
    
    # if st.button("Genetate SweetViz Report"):
    #     st_display_seetviz("SWEETVIZ_REPORT.html")
        
    elif choose_viz == "Correlation":
        fig = plt.figure(figsize=(22, 9))
        sns.heatmap(df.corr(), annot=True, center=True, linewidths=2) 
        st.pyplot(fig)
        
    elif choose_viz == "Line":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.line(df, x=var_x, color=var_color)
        st.plotly_chart(fig)
        
    elif choose_viz == "Histogram":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X variable", numeric_cols)
        var_y = st.sidebar.selectbox("Y variable", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.histogram(df, x=var_x, y=var_y, color=var_color, marginal="rug", hover_data=df.columns)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
        
    elif choose_viz == "Violin":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.violin(df, x=var_x, y=var_y, color=var_color, box=True, points="all", hover_data=df.columns)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
    
    elif choose_viz == "Density":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.density_contour(df, x=var_x, y=var_y)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
        
    elif choose_viz == "Heatmaps":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.density_heatmap(df, x=var_x, y=var_y, marginal_x="rug", marginal_y="histogram")
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
                
    elif choose_viz == "Scatter":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.scatter(data_frame=df, x=var_x, y=var_y, color=var_color, size_max=60)
        st.plotly_chart(fig)
        
    elif choose_viz == "Area":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.area(df, x=var_x, y=var_y, color=var_color, line_group=var_color)
        st.plotly_chart(fig)
        
    elif choose_viz == "Funnel":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X axis", numeric_cols)
        var_y = st.sidebar.selectbox("Y axis", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.funnel(df, x=var_x, y=var_y, color=var_color)
        st.plotly_chart(fig)
        
    elif choose_viz == "Pie":
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_y = st.sidebar.selectbox("Y axis", categories)
        fig = px.pie(values=df[var_y].value_counts(), names=df[var_y].value_counts().index)
        st.plotly_chart(fig)
        
    elif choose_viz == "Box":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X variable", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.box(df, x=var_x, color=var_color, notched=True)
        st.plotly_chart(fig)
    
    elif choose_viz == "Ecdf":
        numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
        categories = df.select_dtypes(include=['object']).columns.to_list()
        var_x = st.sidebar.selectbox("X variable", numeric_cols)
        var_color = st.sidebar.selectbox("Marker", categories)
        fig = px.ecdf(df, x=var_x, color=var_color)
        st.plotly_chart(fig)
# ========== END VISUALIZATION ========== # 

# ========== PREDICTION SYSTEM (MACHINE LEARNING) ========== #
def pred():
    dt = pd.read_csv("dataset/ParisHousing.csv")
    X = dt.drop("price", axis=1)
    y = dt["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        squareMeters = float(st.number_input('squareMeters', min_value=0))
        
    with col2:
        numberOfRooms = int(st.number_input('numberOfRooms', min_value=0))
        
    with col3:
        hasYard = int(st.number_input('hasYard', min_value=0, max_value=1))
        
    with col1:
        hasPool = int(st.number_input('hasPool', min_value=0, max_value=1))
        
    with col2:
        floors = int(st.number_input('floors', min_value=0))
        
    with col3:
        cityCode = int(st.number_input('cityCode', min_value=0))
        
    with col1:
        cityPartRange = int(st.number_input('cityPartRange', min_value=0))
        
    with col2:
        numPrevOwners = int(st.number_input('numPrevOwners', min_value=0))
        
    with col3:
        made = int(st.number_input('made', min_value=1990, max_value=date  ))
        
    with col1:
        isNewBuilt = int(st.number_input('isNewBuilt', min_value=0, max_value=1))
        
    with col2:
        hasStormProtector = int(st.number_input('hasStormProtector', min_value=0, max_value=1))
        
    with col3:
        basement = int(st.number_input('basement', min_value=0))
        
    with col1:
        attic = int(st.number_input('attic', min_value=0))
        
    with col2:
        garage = int(st.number_input('garage', min_value=0))
    
    with col3:
        hasStorageRoom = int(st.number_input('hasStorageRoom', min_value=0, max_value=1))
    
    with col1:
        hasGuestRoom = int(st.number_input('hasGuestRoom', min_value=0))
    
    y_pred = model.predict([[squareMeters, numberOfRooms, hasYard, hasPool, floors,cityCode, cityPartRange, numPrevOwners, made, isNewBuilt,hasStormProtector, basement, attic, garage, hasStorageRoom, hasGuestRoom]]).flatten()[0]

    if st.button('House Price Result'):
        with st.spinner("In progress..."):
            time.sleep(5)
            cfa_total = y_pred * 659.86
            dollars_total = y_pred * 1.10
            st.write("Price prediction:", round(y_pred, 2), "€ |", round(cfa_total, 2), "FCFA |", round(dollars_total, 2), "$")
            st.balloons()
       
    choice = st.sidebar.selectbox("Prediction (Machine Learning Model)", ["Score", "Coef", "Intercept", "PredictionError"])  
    if choice == "Score":
        st.write("Score: ", round(model.score(X_test, y_test)*100), "%")
        
    elif choice == "Coef":
        st.write("Coef: ", model.coef_)
        
    elif choice == "Intercept":
        st.write("Intercept: ", model.intercept_)
        
    elif choice == "PredictionError":
        fig = plt.figure(figsize=(5, 5))
        viz = PredictionError(model)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show();
        st.pyplot(fig)       
# ========== END PREDICTION SYSTEM (MACHINE LEARNING) ========== #
     
# ========== MAIN ========== #
def main():
    st.markdown(hide_menu, unsafe_allow_html=True)
    if selected == "MENU":
        st.write("steamlit-disqus-demo")
    elif selected == "DATASET":
        st.write("***Dataset***")
        st.experimental_data_editor(df)
        
        tr = df.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download data', data=tr, mime='text/csv', file_name='HousingParis.csv')
        
    elif selected == "EDA":
        eda()
    elif selected == "VISUALIZATION":
        st.write("***Visualization***")
        visual()
    elif selected == "PREDICTION":
        st.write("***Prediction system***")
        pred()
# ========== END MAIN ========== #   
   
if __name__ == "__main__":
    main()

# ========== FOOTER ========== #
ligne = """
    <div>
        <hr>
    </div>
"""
st.markdown(ligne, unsafe_allow_html = True)
st.markdown(
            f"""
            <p style="color:#d33682;">&copy; {date} Davila Rostaing, Data Scientist.<p/>
            """
            , unsafe_allow_html = True)
# ========== END FOOTER ========== #