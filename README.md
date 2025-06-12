# Shoulder Redislocation Risk Predictor (ShRed Model)

A Streamlit web application that predicts the risk of shoulder redislocation based on patient characteristics and measurements.

## Features
- Interactive web interface for inputting patient data
- Real-time risk prediction using a pre-trained machine learning model
- Visual representation of patient parameters using a radar chart
- Clear display of prediction results with probability scores

## Requirements
- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- numpy
- plotly

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/shoulder-redislocation-predictor.git
   cd shoulder-redislocation-predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have the pre-trained model file (`ShRed.pkl`) in the project directory
2. Run the application:
   ```bash
   streamlit run main.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
4. Adjust the patient parameters using the sidebar controls
5. View the prediction and probability scores

## Model Information
- Model type: Random Forest Classifier
- Input features: 16 patient parameters
- Output: Binary classification (Yes/No) for redislocation risk with probability scores

## License
This project is for educational and research purposes only. Please consult with a healthcare professional for medical advice.

## Contact
Developed by Dr. Mohamed Esmat  
T&O Cambridge University Hospital  
Email: mohamed.mahmoud19@nhs.net
