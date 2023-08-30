import numpy as np
import pickle
import streamlit as st
import pickle
import sklearn

import pickle

# Load the models from the pickle file
with open('trained_models_Gradient.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

gradboost_model = loaded_models['GradBoost']
# decision_tree_model = loaded_models['DecisionTreeClassifier']



def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.array(input_data,dtype=np.float32)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = gradboost_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Airline Not Delayed'
    else:
        return 'Airline Delayed'


def main():
    # giving a title
    st.title('Airline Delay')

    options = ['Logistic Regression', 'DecisionTreeClassifier']  # Add your model options
    selected_model = st.selectbox('Select a Model', options)

    # getting the input data from the user  DEP_DELAY	DISTANCE	FL_DAYOFWEEK	INDEX_CARRIER	TimeSlot	FL_MONTH	ACTUAL_ELAPSED_TIME	INDEX_ORIGIN	

    DEP_DELAY = st.text_input('Department Delay')
    DISTANCE = st.text_input('Distance')
    FL_DAYOFWEEK = st.text_input('FL day of week')
    INDEX_CARRIER = st.text_input('INDEX_CARRIER')
    TIMESLOT=st.text_input('TIMESLOT')
    FL_MONTH = st.text_input('Month')
    ACTUAL_ELAPSED_TIME = st.text_input('Elapsed Time')
    INDEX_ORIGIN = st.text_input('Index Origin')
    

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Test Result'):
        diagnosis = diabetes_prediction(
            [DEP_DELAY,DISTANCE,FL_DAYOFWEEK,INDEX_CARRIER,TIMESLOT,FL_MONTH,ACTUAL_ELAPSED_TIME,INDEX_ORIGIN])

    st.success(diagnosis)


if __name__ == '__main__':
    main()

##################################################################################


