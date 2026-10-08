# linear-regression

Small project to predict the price of a car from its mileage, using a linear regression trained with gradient descent. No sklearn, everything is done by hand with numpy.

## Setup

```
pip install -r requirements.txt
```

## How to use

First train the model:

```
python3 train_model.py
```

It uses `data.csv` by default, but you can give it another file (needs `mileage` and `price` columns):

```
python3 train_model.py data1.csv
```

This saves the slope and intercept in `model_params.json`.

Then run the prediction:

```
python3 predict.py
```

Type a mileage and it gives you the estimated price, with a plot of the data and the regression line. Type `-1` to quit.

If you run `predict.py` before training, it still works but predicts 0 (the model starts with slope = 0 and intercept = 0).

## How it works

The model is just a line: `price = intercept + slope * mileage`.

Training starts with both values at 0, then for each epoch:
- predict prices with the current line
- compute how wrong we are (mean squared error) and the gradient for slope and intercept
- move both a little in the direction that lowers the error

Mileage and price are scaled between 0 and 1 before training, otherwise the numbers are way too big and gradient descent goes crazy. At the end the parameters are converted back to the real scale.

On `data.csv` the result is basically the same as an exact least squares fit (around -0.0214 for the slope and 8500 for the intercept).

## Files

- `train_model.py`: trains the model and saves it
- `predict.py`: asks for a mileage and shows the prediction
- `data.csv`: the small dataset (23 cars)
- `data1.csv`: a bigger generated dataset (~1000 rows)
