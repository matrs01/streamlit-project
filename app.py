import streamlit as st
import pandas as pd
from paths import PROCESSES_DATA_DIR, ICON_DIR
import eda
from PIL import Image

image = Image.open(ICON_DIR)

st.set_page_config(
    page_title="Customers response to bank offers",
    page_icon=image,
)

st.title('Customers response to bank offers')
st.image(image)


@st.cache_data
def load_data():
    data = pd.read_csv(PROCESSES_DATA_DIR)
    return data


with st.spinner('Loading data...'):
    data = load_data()

# Removing the 'ID_CLIENT' and 'ID_LOAN' columns which are identifiers
df_reduced = data.drop(
    columns=['ID_CLIENT', 'ID_LOAN', 'AGREEMENT_RK'])

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.header('Feature distribution')
st.subheader('Numerical features')
selected_num_features = st.multiselect('Select feature(s)',
                                       ['All', 'PERSONAL_INCOME', 'CREDIT', 'AGE', 'CHILD_TOTAL',
                                        'GENDER', 'DEPENDANTS', 'TERM', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'OWN_AUTO', 'TARGET', 'WORK_TIME'],
                                       default='All')
if st.button("Select", key='num_feature_distr'):
    with st.spinner('Plotting...'):
        st.pyplot(eda.numerical_feature_distribution(
            data, selected_num_features))

st.subheader('Categorical feature')
selected_cat_features = st.multiselect('Select feature(s)',
                                       ['All'] + eda.CATEGORICAL_FEATURES,
                                       default='All')
if st.button("Select", key='cat_feature_distr'):
    with st.spinner('Plotting...'):
        st.pyplot(eda.categorical_feature_distribution(
            data, selected_cat_features))


st.header('Feature correlation')
with st.spinner('Plotting...'):
    st.pyplot(eda.correlation_map(df_reduced))


st.header('Target dependency on features')
with st.spinner('Plotting...'):
    st.pyplot(eda.target_dependency(df_reduced))


st.header('Numerical Characteristics of DataFrame')
st.write(df_reduced.describe())
