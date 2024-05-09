import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# Load the model and other necessary data for the first Excel file
model = pickle.load(open('model_new1.sav', 'rb'))
df1 = pd.read_excel("player_name.xlsx")

# Load the necessary data for the second Excel file
df2 = pd.read_excel("team_name.xlsx")

# Function to search for the most matched row for the first Excel file
def search(name, data):
    max_match = 0
    best_row = None
    
    for index, row in data.iterrows():
        current_match = 0
        for column in data.columns:
            if isinstance(row[column], str) and name.lower() in row[column].lower():
                current_match += 1
        if current_match > max_match:
            max_match = current_match
            best_row = row
    
    return best_row

# Function to display a dictionary in a table on the sidebar
def display_dictionary_in_table_sidebar(dictionary):
    for key, value in dictionary.items():
        st.sidebar.write(f"{key} | {value}")

# Dictionaries to display on the sidebar
my_dict6 = {0: 'All-Rounder', 1: 'Batsman', 2: 'Bowler', 3: 'Wicket Keeper'}
my_dict7 = {0: 'Indian', 1: 'Overseas'}

# Main function to run the Streamlit app

def main():

    # Sidebar for the first search
    st.sidebar.title('Search Player Key number')
    search_input_1 = st.sidebar.text_input("Enter a Player Name to search:")
    if st.sidebar.button("Search Player"):
        if search_input_1:
            result_1 = search(search_input_1, df1)
            if result_1 is not None:
                st.sidebar.write("Here is the Key Number of Player:")
                st.sidebar.write(result_1)
            else:
                st.sidebar.write("This Player is not undergone the auction process in recent years.")
        else:
            st.sidebar.write("Please enter a name to search.")

    # Sidebar for the second search
    st.sidebar.title('Search Team key number')
    search_input_2 = st.sidebar.text_input("Enter a Team Name to search:")
    if st.sidebar.button("Search Team"):
        if search_input_2:
            result_2 = search(search_input_2, df2)
            if result_2 is not None:
                st.sidebar.write("Here is the Key Number of Team:")
                st.sidebar.write(result_2)
            else:
                st.sidebar.write("No match found.")
        else:
            st.sidebar.write("Please enter a name to search.")

        # Display dictionaries on the sidebar
    st.sidebar.title("Role and Key Number:")
    display_dictionary_in_table_sidebar(my_dict6)
    st.sidebar.write("  \n")
    st.sidebar.title("Origin and Key Number:")
    display_dictionary_in_table_sidebar(my_dict7)

    # Main content
    st.title('IPL Player Salary Predictor')
    image = Image.open('iplaml.jpg')
    st.image(image, '')
    st.write("  \n")
    st.markdown("## Enter the player data here to predict the Salary :-")
    st.markdown("###### (You can search the key numbers from the sidebar)")


    # Get user input for prediction
    Player = st.text_input("Enter Player Key Number", )
    Role = st.slider('Enter Role Key Number', 0, 3, 0)
    Team = st.text_input("Enter Team Key Number", )
    Year = st.selectbox(
        'Select a Year',
        ('2025', '2026', '2027')
    )
    Player_Origin = st.slider('Enter Player Origin', 0, 1, 0)

    user_report_data = {
        'Player': Player,
        'Role': Role,
        'Team': Team,
        'Year': Year,
        'Player_Origin': Player_Origin,
    }
    user_data = pd.DataFrame(user_report_data, index=[0])

    st.subheader('Player Data')
    st.write(user_data)

    # Perform prediction
    try:
        salary = model.predict(user_data)
        st.subheader('Player Salary :-')
        st.subheader('₹ ' + str(np.round(salary[0], 2)))
    except ValueError:
        st.write("Please enter Player Data above.")

if __name__ == "__main__":
    main()
