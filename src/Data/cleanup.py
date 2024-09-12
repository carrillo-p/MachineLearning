import pandas as pd
import os

def recode_data():

    for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith("airline.csv")]:
                os.chdir(dirpath)

    df_airline = pd.read_csv('airline_imp.csv')

    df_airline.loc[df_airline['Arrival Delay in Minutes'] < 0, 'Arrival Delay in Minutes'] = 0

    df_airline.loc[df_airline['Gender'] == 'Male', 'Gender'] = 0
    df_airline.loc[df_airline['Gender'] == 'Female', 'Gender'] = 1
    df_airline['Gender'] = df_airline['Gender'].astype('int64')

    df_airline.loc[df_airline['Customer Type'] == 'Loyal Customer', 'Customer Type'] = 0
    df_airline.loc[df_airline['Customer Type'] == 'disloyal Customer', 'Customer Type'] = 1
    df_airline['Customer Type'] = df_airline['Customer Type'].astype('int64')

    df_airline.loc[df_airline['Type of Travel'] == 'Personal Travel', 'Type of Travel'] = 0
    df_airline.loc[df_airline['Type of Travel'] == 'Business travel', 'Type of Travel'] = 1
    df_airline['Type of Travel'] = df_airline['Type of Travel'].astype('int64')

    df_airline.loc[df_airline['Class'] == 'Eco Plus', 'Class'] = 0
    df_airline.loc[df_airline['Class'] == 'Business', 'Class'] = 1
    df_airline.loc[df_airline['Class'] == 'Eco', 'Class'] = 2
    df_airline['Class'] = df_airline['Class'].astype('int64')

    df_airline.loc[df_airline['satisfaction'] == 'neutral or dissatisfied', 'satisfaction'] = 0
    df_airline.loc[df_airline['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    df_airline['satisfaction'] = df_airline['satisfaction'].astype('int64')

    df_airline = df_airline.drop(columns = ['id'])

    df_airline.to_csv('airline_recoded.csv', index = False)


recode_data()


