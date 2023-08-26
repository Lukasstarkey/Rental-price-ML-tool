import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from scipy.stats import norm
import logging



# Configure logging to print debug messages
logging.basicConfig(level=logging.DEBUG)

# Load the saved model
model = joblib.load('/Users/lou/Desktop/Streamlit_app/saved_model2.joblib')

# Load the dataset
df = pd.read_csv('/Users/lou/Desktop/projects/Capstone/capstone data/Lukas Starkey data.csv')

# Convert 'Publish Date' column to datetime format
df['Publish Date'] = pd.to_datetime(df['Publish Date'], format='%d/%m/%Y')

# Create a new column 'Apartment' based on 'Listing Subtype'
df['Apartment'] = df['Listing Subtype'] == 'Apartment'

# Get the unique suburbs from the 'Suburb' column
unique_suburbs = df['Suburb'].unique()

# Drop all other unnecessary columns from the DataFrame
df = df[['Suburb', 'Price', 'No Of Bedrooms', 'BathroomFull', 'Listing Subtype', 'Apartment', 'Publish Date']]

# Drop rows with null values
df.dropna(inplace=True)

# Preprocessing functions

def preprocess_average_price_lag_by_suburb(df, suburb):
    
    # Filter the dataframe based on the user input for suburb
    filtered_df = df[df['Suburb'] == suburb]

    # Calculate the average price for the suburb and three-month window
    filtered_df['Average Price Suburb'] = filtered_df.groupby(pd.Grouper(key='Publish Date', freq='3M'))['Price'].transform('median')

    # Shift the average price by one row to exclude the current listing
    filtered_df['Average Price Lag By Suburb'] = filtered_df['Average Price Suburb'].shift()

    # Replace missing values with the next available value
    filtered_df['Average Price Lag By Suburb'].fillna(method='bfill', inplace=True)

    return filtered_df['Average Price Lag By Suburb'].iloc[0]


def preprocess_average_price_lag_bedrooms(df, bedrooms):
    
    # Filter the dataframe based on the user input for bedrooms
    filtered_df = df[df['No Of Bedrooms'] == bedrooms]

    # Calculate the average price lag
    filtered_df['Average Price Bedrooms'] = filtered_df.groupby([pd.Grouper(key='Publish Date', freq='3M')])['Price'].transform('mean')
    filtered_df['Average Price Lag Bedrooms'] = filtered_df['Average Price Bedrooms'].shift()

    # Replace missing values with the next available value
    filtered_df['Average Price Lag Bedrooms'].fillna(method='bfill', inplace=True)

    return filtered_df['Average Price Lag Bedrooms'].iloc[0]



def preprocess_average_price_lag_bathroomfull(df, bathroom_full):
    
    # Filter the dataframe based on the user input for BathroomFull
    filtered_df = df[df['BathroomFull'] == bathroom_full]

    # Calculate the average price lag
    filtered_df['Average Price BathroomFull'] = filtered_df.groupby([pd.Grouper(key='Publish Date', freq='3M')])['Price'].transform('mean')
    filtered_df['Average Price Lag BathroomFull'] = filtered_df['Average Price BathroomFull'].shift()

    # Replace missing values with the next available value
    filtered_df['Average Price Lag BathroomFull'].fillna(method='bfill', inplace=True)

    return filtered_df['Average Price Lag BathroomFull'].iloc[0]

def preprocess_average_price_lag_apartment(df, apartment):
    
    # Convert apartment to integer (0 for False, 1 for True)
    is_apartment = 1 if apartment else 0

    # Filter the dataframe based on the user input for apartment
    filtered_df = df[df['Apartment'] == is_apartment]

    # Calculate the average price lag
    filtered_df['Average Price Apartment'] = filtered_df.groupby([pd.Grouper(key='Publish Date', freq='3M')])['Price'].transform('mean')
    filtered_df['Average Price Lag Apartment'] = filtered_df['Average Price Apartment'].shift()

    # Replace missing values with the next available value
    filtered_df['Average Price Lag Apartment'].fillna(method='bfill', inplace=True)

    return filtered_df['Average Price Lag Apartment'].iloc[0]




# Define the Streamlit app
def main():
    st.title('Auckland Rental Price Prediction Prototype')

    
    # Define the input fields
    suburb = st.selectbox('Suburb', unique_suburbs)
    
    # Bathroom Ensuites input
    bathroom_ensuites = st.number_input('Number of bathroom ensuites', min_value=0, max_value=10, value=0)
    
    # Bathroom Full input
    bathroom_full = st.number_input('Number of full bathrooms', min_value=1, max_value=10, value=1)
    
    # No Of Bedrooms input
    bedrooms = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=1)
    
    # Apartment input
    apartment = st.selectbox('Apartment', [True, False])

    # Logging the values of suburb, bedrooms, bathroom_full, and apartment
    logging.debug("Suburb: %s", suburb)
    logging.debug("Bedrooms: %s, Bathrooms: %s", bedrooms, bathroom_full)
    logging.debug("Apartment: %s, Bedrooms: %s", apartment, bedrooms)

    if st.button('Make Prediction'):
        # Preprocess 'Average Price Lag By Suburb'
        average_price_lag_by_suburb = preprocess_average_price_lag_by_suburb(df, suburb)

        # Preprocess 'Average Price Lag Bedrooms'
        average_price_lag_bedrooms = preprocess_average_price_lag_bedrooms(df, bedrooms)

        # Preprocess 'Average Price Lag BathroomFull'
        average_price_lag_bathroomfull = preprocess_average_price_lag_bathroomfull(df, bathroom_full)

        # Preprocess 'Average Price Lag Apartment'
        average_price_lag_apartment = preprocess_average_price_lag_apartment(df, apartment)

       # Generate a prediction using the loaded model
        input_data = {
            'Bathroom Ensuites': [bathroom_ensuites],
            'Bathroom Full': [bathroom_full],
            'No Of Bedrooms': [bedrooms],
            'Average Price Lag Bedrooms': [average_price_lag_bedrooms],
            'Average Price Lag By Suburb': [average_price_lag_by_suburb],
            'Average Price Lag Apartment': [average_price_lag_apartment],
            'Average Price Lag BathroomFull': [average_price_lag_bathroomfull]
        }

        # Convert input_data to DataFrame and ensure all columns are numeric
        input_df = pd.DataFrame(input_data)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
  
        # Drop rows with any NaN values
        input_df.dropna(inplace=True)

        if input_df.empty:
            st.write("Input data is empty. Please check your inputs.")
            return  # Return early to avoid further processing
        else:
            # Ensure the input data is 2-dimensional
            input_data_2d = input_df.values

            # Generate a prediction using the loaded model
            prediction = model.predict(input_data_2d)[0]

           # Post-processing: Round the prediction to the nearest 10 and calculate the range
        rounded_prediction = round(prediction / 10) * 10
        lower_bound = rounded_prediction - 50
        upper_bound = rounded_prediction + 50

        # Format the lower and upper bounds with the dollar sign and thousands separators
        formatted_lower_bound = "${:,.0f}".format(lower_bound)
        formatted_upper_bound = "${:,.0f}".format(upper_bound)

        # Display the prediction range with the dollar signs
        st.write('Predicted rental price range:', '$',formatted_lower_bound, '-', '$',formatted_upper_bound)

        if apartment:
            st.code(
                "............................................     ^                   "
                "   ^     ^  ^        _|__|__|_           ^   ^\n"
                "     ___________    _|  | |  |_    ___________   ^\n"
                "    (__IXIXIXIXI___|_|__|_|__|_|___IXIXIXIXI__)\n"
                "    (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "    (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "    (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "    (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "    (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "  /)(__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)\n"
                "_/ )(__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)_/)_\n"
                " ~^^(__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__) ~~^\n"
                "^~~ (__|\"|\"|\"|\"| [=][=] [=] [=][=] |\"|\"|\"|__)~~^\n"
                "\"\"\"\"\"IXI~IXI~IXI~IXI~=I=I=I=I=~IXI~IXI~IXI~IXI\"\"\"\"\"\n"
                "     \"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"|   |\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\""
            )
        else:
            st.code(
                "         _ _\n"
                "        ( Y )\n"
                "         \\ /\n"
                "          \\         /^\\\n"  
                "            )       //^\\\n"
                "         (         //   \\\n"
                "           )      //     \\\n"
                "          __     //       \\\n"
                "         |=^|   //    _    \\\n"
                "       __|= |__//    (+)    \\\n"
                "      /LLLLLLL//      ~      \\\n"
                "     /LLLLLLL//               \\\n"
                "    /LLLLLLL//                 \\\n"
                "   /LLLLLLL//  |~[|]~| |~[|]~|  \\\n"
                "   ^| [|] //   | [|] | | [|] |   \\\n"
                "    | [|] ^|   |_[|]_| |_[|]_|   |^\n"
                " ___|______|                     |\n"
                "/LLLLLLLLLL|_____________________|\n"
                "/LLLLLLLLLLL/LLLLLLLLLLLLLLLLLLLLLL\\\n"
                "/LLLLLLLLLLL/LLLLLLLLLLLLLLLLLLLLLLLL\\\n"
                "^||^^^^^^^^/LLLLLLLLLLLLLLLLLLLLLLLLLL\\\n"
                " || |~[|]~|^^||^^^^^^^^^^||^|~[|]~|^||^^\n"
                " || | [|] |  ||  |~~~~|  || | [|] | ||\n"
                " || |_[|]_|  ||  | [] |  || |_[|]_| ||\n"
                " ||__________||  |   o|  ||_________||\n"
                ".'||][][][][][|| | [] |  ||[][][][][||.'.\n"
                ".'||[][][][][]||_-`----'-_||][][][][]||'.\"\n"
                ".'(')^(.)(').( )'^@/-- -- - --\\@( )'(.),( ).(').\n"
            )


if __name__ == '__main__':
    main()
