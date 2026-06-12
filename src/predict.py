import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_model_and_scaler():
    model  = joblib.load(os.path.join(MODELS_PATH, "RandomForest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    print("Model and scaler loaded")
    return model, scaler


def prepare_input(airline, source_city, departure_time, stops,
                  arrival_time, destination_city, flight_class,
                  duration, days_left):

    
    stops_map  = {"zero": 0, "one": 1, "two_or_more": 2}
    class_map  = {"Economy": 0, "Business": 1}

    
    airline_map       = {"AirAsia": 0, "Air_India": 1, "GO_FIRST": 2,
                         "Indigo": 3, "SpiceJet": 4, "Vistara": 5}
    city_map          = {"Bangalore": 0, "Chennai": 1, "Delhi": 2,
                         "Hyderabad": 3, "Kolkata": 4, "Mumbai": 5}
    time_map          = {"Afternoon": 0, "Early_morning": 1, "Evening":2,"Late_Night": 3,
                         "Morning": 4, "Night": 5}

    input_data = {
        "airline":          airline_map[airline],
        "source_city":      city_map[source_city],
        "departure_time":   time_map[departure_time],
        "stops":            stops_map[stops],
        "arrival_time":     time_map[arrival_time],
        "destination_city": city_map[destination_city],
        "class":            class_map[flight_class],
        "duration":         duration,
        "days_left":        days_left,
    }

    return pd.DataFrame([input_data])


def predict_price(airline, source_city, departure_time, stops,
                  arrival_time, destination_city, flight_class,
                  duration, days_left):

    model, scaler = load_model_and_scaler()
    input_df      = prepare_input(airline, source_city, departure_time, stops,
                                  arrival_time, destination_city, flight_class,
                                  duration, days_left)
    input_scaled  = scaler.transform(input_df.values)
    prediction    = model.predict(input_scaled)[0]
    print(f"Predicted price: ₹{prediction:,.0f}")
    return prediction


if __name__ == "__main__":
    predict_price(
        airline          = "Indigo",
        source_city      = "Delhi",
        departure_time   = "Morning",
        stops            = "one",
        arrival_time     = "Evening",
        destination_city = "Mumbai",
        flight_class     = "Economy",
        duration         = 3.5,
        days_left        = 15
    )