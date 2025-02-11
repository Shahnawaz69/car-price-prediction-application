
# Car Price Prediction App

This is a simple Streamlit application that predicts car prices (MPG - Miles Per Gallon) based on various car features using a linear regression model.

## Features
- Loads a car dataset from Seaborn's dataset repository.
- Preprocesses categorical data using Label Encoding.
- Trains a linear regression model to predict MPG.
- Displays model evaluation metrics (Mean Squared Error, R² Score).
- Allows users to input custom data and get a prediction.

## Technologies
- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For creating and evaluating the linear regression model.
- **Pandas**: For handling and processing the dataset.
- **NumPy**: For numerical operations.

## Requirements

Make sure to install the necessary Python packages before running the app. You can install them by running:

```
pip install -r requirements.txt
```

The required packages are:
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`

## How to Run

1. Clone or download the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run car_price_prediction_app.py
   ```
4. Open your web browser and go to the provided URL (typically `http://localhost:8501`).
5. Use the input form to enter data and get a car price prediction.

## License

This project is open-source and available under the [MIT License](LICENSE).
