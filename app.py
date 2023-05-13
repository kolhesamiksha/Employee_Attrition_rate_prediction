import streamlit as st
from streamlit_option_menu import option_menu
import os
import requests
from PIL import Image, ImageOps
from io import BytesIO
import utils as ut
import numpy as np
import time
import pandas as pd
import plotly.express as px
import pickle
import hydralit_components as hc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

routes = os.environ["ROUTE"]

response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(
    page_title='Attrition_rate_prediction_app', 
    page_icon = im, 
)

def draw_something_on_top_of_page_navigation():
    st.sidebar.image('katonic_logo.png')

draw_something_on_top_of_page_navigation()

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["About","Dashboard","App"]
    )

if selected == "About":
    st.write("""
        # Employee Attrition rate prediction 
        """)
    st.write("##### The Attrition rate helps an Organisation to find their progress over a specific period")
    img1 = Image.open('employee_attrition.png')
    st.image(img1,use_column_width=True)
    st.write("""###### The employee attrition rate is the measure of people who leaves the organization. By measuring the attrition rate, we can identify the causes and factors that need to be solved to eliminate employee attrition""")
    st.write("""###### Employee attrition is expressed as the normal process by which the employees leave the organization due to some Social, Personal, Finantial,Professional factors.Employee Atrrition rate monitoring term is more familiar to HR professionals.""")
                       
    st.write("###### attrition rate detection is critical to organizations because it helps them to reduce costs, plan for the future, retain valuable employees, boost employee morale, and gain a competitive advantage.")

    st.write("###### depending upon the reasons the Types of Attributions are as follows:")
    st.write("""1. **Voluntary attrition**: when employee leaves the organisation due to personal reasons.""")
    st.write("""2. **Involuntary attrition**: when the organization ends the employment process due to some internal/market conditions""")
    st.write("""3. **External attrition**: when an employee leaves an organization to work for another organization.""")
    st.write("""4. **Internal attrition**: when an employee is given another position within the same organization as a promotion.""")
    
    st.success("Visit Dashboard to conclude the key factors of Attrition.")

data = pd.read_csv('train.csv')

fig2_chart = pd.crosstab(data['YearsAtCompany'],data['Attrition'])
fig2_chart = fig2_chart[:10]

df_target = data[['id', 'Attrition']].groupby('Attrition').count()
fig1_chart = go.Figure(data=[go.Pie(values=df_target['id'],
                                    hole=.5)])

fig1_chart.update_layout(showlegend=True)
fig1_chart.update_traces(textposition='inside')

f_df = data[data['Gender']=='Female']
m_df = data[data['Gender']=='Male']

cst_f = pd.crosstab(f_df['Age'],f_df['Attrition'])
cst_m = pd.crosstab(m_df['Age'],m_df['Attrition'])

fig3_chart = pd.crosstab(data['YearsInCurrentRole'],data['Attrition'])
fig3_chart = fig3_chart[:10]

fig4_chart = pd.crosstab(data['YearsWithCurrManager'],data['Attrition'])
fig4_chart = fig4_chart[:10]

fig7_data = data[data['Attrition']== 1]
fig7_chart =  pd.crosstab(fig7_data['JobRole'],fig7_data['Attrition'])


if selected == "Dashboard":
    st.title("Employee Attrition Analysis Dashboard📊")
    if st.button("Live Dashboard"):
        with hc.HyLoader("Analysing",hc.Loaders.pretty_loaders,index=[3]):
            time.sleep(5)

        placeholder = st.empty()
        with placeholder.container():
            fig_col1, fig_col2, fig_col3, fig_col4 = st.columns(4)
            size_fixer = plt.figure(figsize = (3,3))
            
            st.markdown("#### Employee Attrition")
            fig1 = st.plotly_chart(fig1_chart)

            st.markdown("### Years with company")
            fig2 = st.bar_chart(fig2_chart)
             
            st.markdown("### Year In Current Company")
            fig3 = st.area_chart(fig3_chart)

            plt.subplot(3, 3, 4);
            st.markdown("### Year with Current Manager")
            fig4 = st.area_chart(fig4_chart)

            st.markdown("### Attrition Age Distribution by Gender Female")
            fig5 = st.area_chart(cst_f)

            st.markdown("### Attrition Age Distribution by Gender Male")
            fig6 = st.area_chart(cst_m)

            st.markdown("### Attrition by job role")
            fig7 =st.bar_chart(fig7_chart)

        st.success("Visit App to predict the Attrition status of your Organisation")

file = open("new_rgb_model.pickle","rb")
model = pickle.load(file)

if selected == "App":
    st.write("# Attrition Rate Prediction App")
    st.write("This App will help HR Professionals to manage their ")

    Age = st.slider('Age',0,60)
    
    edu_display = ('Below College','College','Bachelor','Master','Doctor')
    edu_options = list(range(len(edu_display)))
    Education = st.selectbox('Qualification',edu_options,format_func=lambda x: edu_display[x])

    gen_display = ('Female','Male')
    gen_options = list(range(len(gen_display)))
    Gender = st.selectbox('Gender',gen_options,format_func=lambda x: gen_display[x])

    YearsAtCompany = st.slider('Years At Company',0,42)

    MonthlyIncome = st.number_input("Applicant's Monthly Income in k",value=0)

    HourlyRate = st.slider('HourlyRate',30,100)

    JobLevel = st.slider('Job level at organisation',1,7)

    JobSatisfaction = st.slider('Job Satisfaction ratings',1,4)

    JobInvolvement = st.slider('Job Involvement ratings',1,4)


    if st.button("Submit"):
        import hydralit_components as hc
        with hc.HyLoader("submitting",hc.Loaders.pretty_loaders,index=[5]):
            time.sleep(3)

        features = [[Age,Education,Gender,YearsAtCompany,MonthlyIncome,HourlyRate,JobLevel,JobSatisfaction,JobInvolvement]]
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans==0:
            st.success(
                "Congratulations!! The Employee is Happy with You"
                ":smiley:"
            )
        else:
            st.error(
                "Oops! This Employee is likely to be Churn"
                ":slightly_frowning_face:"
            )



            


