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

df_role = pd.read_excel("role.xlsx")
df_origin = pd.read_excel("origin.xlsx")


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




# Main function to run the Streamlit app
def main():
    

    
    st.title('IPL Player Salary Predictor')
    image = Image.open('iplaml.jpg')
    st.image(image, '')
    st.write("  \n")
    st.markdown("## Enter the player data here to predict the Salary :-")
    #st.markdown("###### (You can search the key numbers from the sidebar)")

    # Get user input for prediction
    player_name = st.text_input("Enter Player Name")

    # Dropdown for Role
    role_options = df_role['Role'].tolist()
    selected_role = st.selectbox('Select Role', role_options)
    role_key = df_role[df_role['Role'] == selected_role]['Key Number'].values[0]
    
    # Dropdown for team
    team_options = df2['Team Name'].tolist()
    selected_team = st.selectbox('Select Team', team_options)
    team_key = df2[df2['Team Name'] == selected_team]['Key Number'].values[0]

    Year = st.selectbox(
        'Select a Year',
        ('2025', '2026', '2027')
    )
    
    # Dropdown for Origin
    origin_options = df_origin['Origin'].tolist()
    selected_origin = st.selectbox('Select Origin', origin_options)
    origin_key = df_origin[df_origin['Origin'] == selected_origin]['Key Number'].values[0]

    if player_name:
        player_info = search(player_name, df1)
        if player_info is not None:
            player_key = player_info['Key Number']  # Assuming 'Key' is the column name for the key number
        else:
            st.write("This player has not gone through auction process in recent years.")
            return
    else:
        st.write("Please enter player data.")
        return

   
    

    user_report_data = {
        'Player': player_key,
        'Role': role_key,
        'Team': team_key,
        'Year': Year,
        'Player_Origin': origin_key,
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
