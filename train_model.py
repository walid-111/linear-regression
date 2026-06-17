import sys
import json
import pandas as pd
import numpy as np

"""
Linear Regression finds the best-fitting line (by finding optimal slope and intercept) through our data points using
Gradient Descent algorithm, which repeatedly adjusts these parameters to minimize the Mean Squared Error
(difference between predicted and actual values squared) until we can't improve much further.
"""

class CarPricePredictor:
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
        self.model = {'slope': 0, 'intercept': 0}
        
    def load_data(self, file_path):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: Could not find {file_path}")
            exit()

    def normalize_data(self, values):
        """
        Why scale to range [0,1]?
        1. Makes training more stable
        2. Prevents big numbers from having too much influence
        3. Helps gradient descent converge better

        How it works:
        - For each value: (value - min) / (max - min)
        - This makes smallest value = 0 and largest = 1
        """
        min_val = min(values)
        max_val = max(values)
        return (values - min_val) / (max_val - min_val), min_val, max_val

    def train_model(self, learning_rate=0.1, epochs=1000):
        """
        GRADIENT DESCENT EXPLAINED:
        --------------------------
        - Goal: find the bottom of a valley blindfolded
        -> take small steps in the direction that reduces error
        -> repeat until limit
    	--------------------------
    	The Math Behind It:
		a) Error Calculation (Mean Squared Error):
			error = (actual_price - predicted_price)²
			- We square it to:
			* Make all errors positive (being too high or too low is bad)
			* Punish big errors more than small ones

		b) Gradient Calculation:
			- Take derivative of error with respect to each parameter
			- Tells us how to adjust parameters to reduce error
			- For squared error (x²), derivative is 2x
			- We multiply by -1 to go downhill (minimize)
			
		c) Update Rules:
			slope = slope - learning_rate * slope_gradient
			intercept = intercept - learning_rate * intercept_gradient
			
			If prediction too high:
			- Error is negative
			- Gradient = -2 * negative = positive
			- Subtract positive = decrease parameter
			
			If prediction too low:
			- Error is positive
			- Gradient = -2 * positive = negative
			- Subtract negative = add positive = increase parameter

        Parameters:
        - learning_rate: Size of steps we take
        - epochs: Number of iterations to train
        """
        mileage = self.data['mileage'].values
        prices = self.data['price'].values
        
        norm_mileage, mile_min, mile_max = self.normalize_data(mileage)
        norm_prices, price_min, price_max = self.normalize_data(prices)
        
        slope = 0
        intercept = 0
        
        # Gradient descent
        for _ in range(epochs):
            # 1. Make predictions with current line
            predictions = intercept + slope * norm_mileage
            
            # 2. Calculate error gradients
            # For intercept: average error
            intercept_gradient = -2 * np.mean(norm_prices - predictions)
            # For slope: error weighted by mileage
            slope_gradient = -2 * np.mean((norm_prices - predictions) * norm_mileage)
            
            # 3. Update parameters (take a step down the gradient)
            intercept -= learning_rate * intercept_gradient
            slope -= learning_rate * slope_gradient
        
        # Convert parameters back to original scale
        self.model['slope'] = slope * (price_max - price_min) / (mile_max - mile_min)
        self.model['intercept'] = (intercept * (price_max - price_min) + price_min -  self.model['slope'] * mile_min)

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    predictor = CarPricePredictor(data_file)
    predictor.train_model()
    
    # Save the model parameters (+ which csv was used, so predict.py plots the same data)
    with open('model_params.json', 'w') as f:
        json.dump({**predictor.model, 'data_file': data_file}, f)
    print("Model trained and saved!")

if __name__ == "__main__":
    main()