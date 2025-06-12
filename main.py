import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# Data cleaning function removed - not needed


def add_sidebar():
  st.sidebar.header("Patient Measurements")
  
  slider_labels = [
        ("Age", "Age"),
        ("White British Ethnicity", "White British Ethnicity"),
        ("Physical job", "Physical job"),
        ("Athlete/contact sports", "Athlete/contact sports"),
        ("MeanSurface Glenoid", "MeanSurface Glenoid"),
        ("MeanSurface Humeral", "MeanSurface Humeral"),
        ("Age of first dislocation", "Age of first dislocation"),
        ("Caused by Trauma", "Caused by Trauma"),
        ("Gender_F", "Gender_F"),
        ("Gender_M", "Gender_M"),
        ("Laterality_L", "Laterality_L"),
        ("Laterality_R", "Laterality_R"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    # Handle binary/categorical variables with Yes/No
    if key in ["White British Ethnicity", "Physical job", "Athlete/contact sports", 
               "Caused by Trauma", "Gender_F", "Gender_M", 
               "Laterality_L", "Laterality_R"]:
      input_dict[key] = st.sidebar.selectbox(
        label,
        options=[0, 1],
        index=0,
        format_func=lambda x: "Yes" if x == 1 else "No"
      )
    else:
      # Handle special cases for MeanSurface values
      if key in ["MeanSurface Glenoid", "MeanSurface Humeral"]:
        input_dict[key] = st.sidebar.text_input(
          label,
          value="5.0",
          help="Enter a numeric value"
        )
        # Add validation to ensure numeric input
        if not input_dict[key].replace('.', '', 1).isdigit():
          st.sidebar.error(f"Please enter a valid number for {label}")
          input_dict[key] = "5.0"  # Reset to default
      # Handle Age and Age of first dislocation with 1-100 range
      elif key in ["Age", "Age of first dislocation"]:
        input_dict[key] = st.sidebar.slider(
          label,
          min_value=1,
          max_value=100,
          value=30
        )
      # Default slider for other numeric values
      else:
        input_dict[key] = st.sidebar.slider(
          label,
          min_value=0,
          max_value=10,
          value=5
        )
  
  # Calculate derived values automatically
  input_dict["High activity risk"] = input_dict["Physical job"] + input_dict["Athlete/contact sports"]
  
  try:
    glenoid = float(input_dict["MeanSurface Glenoid"])
    humeral = float(input_dict["MeanSurface Humeral"])
    input_dict["MeanSurface difference"] = glenoid - humeral
    
    # Calculate Bony constraint ratio with division by zero check
    if humeral != 0:
      input_dict["Bony constraint ratio"] = glenoid / humeral
    else:
      input_dict["Bony constraint ratio"] = 0
  except ValueError:
    # If conversion fails, set default values
    input_dict["MeanSurface difference"] = 0
    input_dict["Bony constraint ratio"] = 0
  
  input_dict["Years since first dislocation"] = input_dict["Age"] - input_dict["Age of first dislocation"]
  
  # Display calculated values for user reference
  st.sidebar.subheader("Calculated Values")
  st.sidebar.write(f"High Activity Risk: {input_dict['High activity risk']}")
  st.sidebar.write(f"MeanSurface Difference: {input_dict['MeanSurface difference']:.2f}")
  st.sidebar.write(f"Bony Constraint Ratio: {input_dict['Bony constraint ratio']:.2f}")
  st.sidebar.write(f"Years Since First Dislocation: {input_dict['Years since first dislocation']}")
    
  return input_dict


# Data scaling function removed - not needed
  

def get_radar_chart(input_data):
  
  # Use input data directly - no scaling needed
  
  # Define categories
  categories = ['Age', 'Ethnicity', 'Physical Job', 'Athlete/Sports', 
                'High Activity Risk', 'Glenoid Surface', 'Humeral Surface', 
                'Surface Difference', 'Bony Constraint', 'Age First Dislocation',
                'Years Since First', 'Trauma Caused',
                'Gender F', 'Gender M', 'Laterality L', 'Laterality R']
    
  # Get values in the correct order
  values = [
      input_data.get('Age', 0),
      input_data.get('White British Ethnicity', 0),
      input_data.get('Physical job', 0),
      input_data.get('Athlete/contact sports', 0),
      input_data.get('High activity risk', 0),
      input_data.get('MeanSurface Glenoid', 0),
      input_data.get('MeanSurface Humeral', 0),
      input_data.get('MeanSurface difference', 0),
      input_data.get('Bony constraint ratio', 0),
      input_data.get('Age of first dislocation', 0),
      input_data.get('Years since first dislocation', 0),
      input_data.get('Caused by Trauma', 0),
      input_data.get('Gender_F', 0),
      input_data.get('Gender_M', 0),
      input_data.get('Laterality_L', 0),
      input_data.get('Laterality_R', 0)
  ]
    
  # Create radar chart with colored segments
  fig = go.Figure()
  
  # Color palette for segments
  colors = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
      '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
      '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
  ]
  
  # Add each segment with its own color
  for i in range(len(categories)):
      next_idx = (i + 1) % len(categories)
      fig.add_trace(go.Scatterpolar(
          r=[0, values[i], values[next_idx], 0],
          theta=[categories[i], categories[i], categories[next_idx], categories[i]],
          fill='toself',
          fillcolor=colors[i % len(colors)],
          line=dict(color='white', width=1),
          opacity=0.7,
          showlegend=False
      ))
  
  # Add main outline
  fig.add_trace(go.Scatterpolar(
      r=values + [values[0]],
      theta=categories + [categories[0]],
      line=dict(color='#2c3e50', width=2),
      fill='none',
      showlegend=False
  ))
    
  # Update layout with better styling
  fig.update_layout(
      polar=dict(
          bgcolor='#f8f9fa',
          radialaxis=dict(
              visible=True,
              range=[0, 10],
              gridcolor='white',
              linecolor='gray',
              linewidth=1,
              showline=True,
              tickfont=dict(color='#555')
          ),
          angularaxis=dict(
              linecolor='gray',
              linewidth=1,
              rotation=90
          )
      ),
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      margin=dict(l=40, r=40, t=40, b=40),
      showlegend=False
  )
    
  return fig


def add_predictions(input_data):
  try:
    # Load the model
    model = pickle.load(open("ShRed.pkl", "rb"))
    
    # Prepare input data and add dummy column (17th feature) with value 0
    input_values = list(input_data.values()) + [0.0]  # Add dummy column
    input_array = np.array([input_values])
    
    # Get prediction and probabilities
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    
    # Ensure we're using the correct class indices (0=No, 1=Yes)
    prob_no = probabilities[0]  # Probability of class 0 (No)
    prob_yes = probabilities[1]  # Probability of class 1 (Yes)
    
    st.subheader("Shoulder Redislocation Prediction")
    st.write("The prediction is:")
    
    # Display prediction with consistent probability
    if prediction == 0:
      st.write("<span class='diagnosis no-risk'>No</span>", unsafe_allow_html=True)
    else:
      st.write("<span class='diagnosis high-risk'>Yes</span>", unsafe_allow_html=True)
    
    # Display both probabilities for clarity
    st.write(f"<div style='margin-top: 20px;'><b>Probability of no redislocation:</b> "
             f"<span style='color: #4CAF50; font-weight: bold;'>{prob_no:.2f}</span></div>", 
             unsafe_allow_html=True)
    st.write(f"<b>Probability of redislocation:</b> "
             f"<span style='color: #f44336; font-weight: bold;'>{prob_yes:.2f}</span>", 
             unsafe_allow_html=True)
    
    st.write("<div style='margin-top: 20px; font-size: 0.9em; color: #666;'>"
             "This app is designed to assist medical professionals in assessing shoulder redislocation risk, "
             "but should not be used as a substitute for professional medical evaluation.</div>", 
             unsafe_allow_html=True)
    
  except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
    st.write("Please ensure the model file is properly loaded and the input data matches the model's expected format.")


def main():
  st.set_page_config(
    page_title="Shoulder Redislocation Risk Predictor",
    page_icon=":bone:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.markdown("<h1><span class='highlight'>ShRed</span> Model</h1>", unsafe_allow_html=True)
    st.write("This app predicts shoulder redislocation risk using a machine learning model based on patient characteristics and measurements. You can adjust the patient parameters using the sliders and selectors in the sidebar to see how different factors influence the prediction.")
    
    st.write("Developed by Dr Mohamed Esmat")
    st.write("T&O Cambridge University Hospital")
    st.write("Email: mohamed.mahmoud19@nhs.net")
  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()
 