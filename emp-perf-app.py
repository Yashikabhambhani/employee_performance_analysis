import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import *

st.write("""
# SmartPerf - AI-Powered Employee Insights 
This app predicts the **Employee Performance** based on the various factors!
""")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])
if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)

dataset.drop(['EmpNumber'],inplace=True,axis=1)

# Encoding all the ordinal columns and creating a dummy variable for them to see if there are any effects on Performance Rating
#le = LabelEncoder()
#for i in (2,3,4,5,6,7,16,26):
    #dataset.iloc[:,i] = le.fit_transform(dataset.iloc[:,i])

le = LabelEncoder()
dataset["empdep"] = le.fit_transform(dataset["EmpDepartment"])
dataset["empjr"] = le.fit_transform(dataset["EmpJobRole"])


st.sidebar.header('User Input Parameters')

def user_input_features():
    #age = st.sidebar.slider('Age', 18, 60, 30)
    #gender = st.radio("Gender", data.Gender.unique())
    #edubkgrnd = st.radio("Educational Background", data.EducationBackground.unique())
    empdep = st.sidebar.radio("Department", dataset.EmpDepartment.unique())
    empjr = st.sidebar.radio("Job Role", dataset.EmpJobRole.unique())
    empenvsat = st.sidebar.slider('Environment Satisfaction', 1, 4, 2)
    empsalhike = st.sidebar.slider('Salary Hike Percent', 11, 25, 15)
    empwrklb = st.sidebar.slider('Work Life Balance', 2, 4, 2)
    empyrsatcomp = st.sidebar.slider('Experience Years at this Company',0, 36, 2)
    empyrscurrole = st.sidebar.slider('Experience Years in Current Role', 0, 15, 5)
    empyrspromo = st.sidebar.slider('Years since last Promotion', 0, 15, 2)
    empyrscurrmng = st.sidebar.slider('Years with Current Manager', 0, 17, 2)
    
    data = {'EmpDepartment': empdep,
            'EmpJobRole': empjr,
            'EmpEnvironmentSatisfaction': empenvsat,
            'EmpLastSalaryHikePercent': empsalhike,
            'EmpWorkLifeBalance' : empwrklb,
            'ExperienceYearsAtThisCompany' : empyrsatcomp,
            'ExperienceYearsInCurrentRole' : empyrscurrole,
            'YearsSinceLastPromotion' : empyrspromo,
            'YearsWithCurrManager' : empyrscurrmng }
    features = pd.DataFrame(data, index=[0])
    return features



df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

if(df.EmpDepartment is "Data Science"):
    df.EmpDepartment = 0
        
elif (df.EmpDepartment is "Development"):
    df.EmpDepartment = 1
        
elif (df.EmpDepartment is "Finance"):
    df.EmpDepartment = 2
        
elif (df.EmpDepartment is "Human Resources"):
    df.EmpDepartment = 3
        
elif (df.EmpDepartment is "Research & Development"):
    df.EmpDepartment = 4
    
else:
    df.EmpDepartment = 5


if(df.EmpJobRole is "Business Analyst"):
    df.EmpJobRole = 0
elif (df.EmpJobRole is "Data Scientist"):
    df.EmpJobRole = 1
elif (df.EmpJobRole is "Delivery Manager"):
    df.EmpJobRole = 2
elif (df.EmpJobRole is "Developer"):
    df.EmpJobRole = 3
elif (df.EmpJobRole is "Finance Manager"):
    df.EmpJobRole = 4
elif (df.EmpJobRole is "HealthCare Representative"):
    df.EmpJobRole = 5
elif (df.EmpJobRole is "Human Resources"):
    df.EmpJobRole = 6
elif (df.EmpJobRole is "Laboratory Technician"):
    df.EmpJobRole = 7
elif (df.EmpJobRole is "Manager"):
    df.EmpJobRole = 8
elif (df.EmpJobRole is "Manager R&D"):
    df.EmpJobRole = 9
elif (df.EmpJobRole is "Manufacturing Director"):
    df.EmpJobRole = 10
elif (df.EmpJobRole is "Research Director"):
    df.EmpJobRole = 11
elif (df.EmpJobRole is "Research Scientist"):
    df.EmpJobRole = 12
elif (df.EmpJobRole is "Sales Executive"):
    df.EmpJobRole = 13
elif (df.EmpJobRole is "Sales Representative"):
    df.EmpJobRole = 14
elif (df.EmpJobRole is "Senior Developer"):
    df.EmpJobRole = 15
elif (df.EmpJobRole is "Senior Manager R&D"):
    df.EmpJobRole = 16
elif (df.EmpJobRole is "Technical Architect"):
    df.EmpJobRole = 17
else:
    df.EmpJobRole = 18

X = dataset.iloc[:,[27,28,9,16,20,21,22,23,24]]
Y = dataset.PerformanceRating



clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)



st.subheader('Class labels and their corresponding index number')
p = dataset.PerformanceRating.unique()
p.sort()
st.write(p)

st.subheader('Prediction')
#st.write(dataset.PerformanceRating[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
