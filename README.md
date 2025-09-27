House Price Predictor

A beginner friendly Machine Learning project that predicts housing prices using Linear Regression on the California Housing dataset from scikit-learn. This project demonstrates the end-to-end ML workflow: data preprocessing, model training, evaluation, and visualization.

Features

Loads the California Housing dataset 1990 US Census data.

Trains a Linear Regression model using scikit-learn.

Evaluates performance with Mean Squared Error (MSE) and R² score.

Visualizes actual vs. predicted house prices with a scatter plot.

Tech Stack

Language: Python

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn

Tools: VS Code, Git, GitHub

Project Structure
House Price Predictor/
house_price_predictor.py   Main script
README.md                  Project documentation
venv/                      Virtual environment (ignored in Git)

Installation & Usage

Clone the repository:

git clone https://github.com/Learnlife001/house-price-predictor.git
cd house-price-predictor

Create a virtual environment & activate it:

python -m venv venv
.\venv\Scripts\activate   # On Windows

Install dependencies:

pip install -r requirements.txt

Run the script:

python house_price_predictor.py

Example Output
Training data shape: (16512, 8)
Testing data shape: (4128, 8)
Model training complete.
Mean Squared Error: 0.55
R2 Score: 0.57

Scatter plot of actual vs. predicted prices

![House Price Prediction Plot](<house price predictor.jpg>)

Future Improvements

Add user input to predict custom house prices.

Deploy as a web app with Flask/FastAPI.

Use a more recent dataset (e.g., Kaggle House Prices).

License

This project is open source and free to use under the MIT License.