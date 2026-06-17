import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

class CarPricePredictor:
    def __init__(self, model_file='model_params.json', data_file='data.csv'):
        try:
            with open(model_file, 'r') as f:
                self.model = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("No trained model found, run train_model.py first (using 0 for now)")
            self.model = {'slope': 0, 'intercept': 0}
        self.data = pd.read_csv(self.model.get('data_file', data_file))
    
    def predict_price(self, mileage):
        """Predict car price using trained model."""
        predicted_price = self.model['intercept'] + self.model['slope'] * mileage
        return max(0, predicted_price)  # Ensure price is not negative
    
    def plot_results(self):
        """Visualize data & prediction line"""
        plt.figure(figsize=(10, 6))
        
        # Plot real data points
        plt.scatter(self.data['mileage'], self.data['price'], 
                   color='blue', alpha=0.5, label='Actual Prices')
        # Plot prediction line
        mileage_range = np.linspace(min(self.data['mileage']), max(self.data['mileage']), 100)
        predicted_prices = [self.predict_price(m) for m in mileage_range]
        plt.plot(mileage_range, predicted_prices, 'r-', label='Prediction Line')
        
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Car Price vs Mileage')
        plt.legend()
        plt.grid(True)

def main():
    predictor = CarPricePredictor()
    
    while True:
        mileage_input = input("\nEnter car mileage (or -1 to quit): ")
        
        try:
            mileage = float(mileage_input)
            
            if mileage == -1:
                print("Goodbye!")
                break
                
            if mileage < 0:
                print("Please enter a valid mileage (≥ 0)")
                continue
                
            predicted_price = predictor.predict_price(mileage)
            print(f"\nPredicted car price: ${predicted_price:,.2f}")

            plt.close('all')
            predictor.plot_results()
            plt.show(block=True)
            
        except ValueError:
            print("Please enter a valid number")

if __name__ == "__main__":
    main()