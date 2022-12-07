# Test App, die wir im Kurs zusammen erstellt haben

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


st.set_page_config(page_title="Credit Default App", 
                   page_icon="🦈",
                   layout="wide")

# define load function
@st.cache()
def load_data():
    data=pd.read_csv("prosper_data_app_dev_copy.csv")
    return(data.dropna())

@st.cache()
def load_model():
    filename="finalized_default_model.sav"
    loaded_model = pickle.load(open(filename,"rb"))
    return(loaded_model)

data = load_data()
model = load_model()

st.title("Sharky's Credit Default App")

st.write("🦈 This is Sharky's credity default app. This application is a dashboard to *visualise* credit defaults and to make **predictions**.🦈")



### Section 1 of the app

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interest the customer has to pay",
                data["borrower_rate"].min(),
                data["borrower_rate"].max(),
                (0.05,0.15))

row1_col1.write(rate)


income = row1_col2.slider("Monthly income of customers",
                data["monthly_income"].min(),
                data["monthly_income"].max(),
                (2000.00,30000.00))

row1_col2.write(income)


mask = ~data.columns.isin(["loan_default", "borrower_rate", "employment_status"])
names = data.loc[:, mask].columns
variable = row1_col3.selectbox("Select Variable to Compare", names)



filtered_data = data.loc[(data["borrower_rate"]>=rate[0])&
                       (data["borrower_rate"]<=rate[1])&
                       (data["monthly_income"]>=income[0])&
                       (data["monthly_income"]<=income[1]),
                       :]

if st.checkbox("show filtered data", False):
    st.subheader("raw data")
    st.write(filtered_data)

row2_col1, row2_col2 = st.columns([1,1])

# Create a standard matplotlib barchart 
barplotdata = filtered_data[["loan_default", variable]].groupby("loan_default").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color = "#fc8d62")
ax.set_ylabel(variable)

# Put matplotlib figure in col 1 
row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1, use_container_width=True)


# Create a seaborn figure
fig2 = sns.lmplot(y="borrower_rate", x=variable, data = filtered_data, 
                  order=2, height=4, aspect=1/1, col="loan_default", hue="loan_default",
                  palette = "Set2")

# Put seaborn figure in col 2
row2_col2.subheader("Borrower Rate Correlations")
row2_col2.pyplot(fig2, use_container_width = True)


st.header("Predicting Customer Default")

uploaded_data = st.file_uploader("Choose a file with customer data for making predictions.")

if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers,drop_first=True)
    st.write(new_customers)
    
    predictions = model.predict(new_customers)
    new_customers["predictions"] = predictions
    
    st.download_button(label="Download Scored Customer Data",
                       data=new_customers.to_csv().encode("utf-8"),
                       file_name="scored_new_customers.csv")
