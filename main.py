# -*- coding: utf-8 -*-
"""
Stock Market Prediction Web App
✨ Final Stylish Version for StockScope — Nov 2025 ✨
Now with interactive graphs (Plotly), login system, and clean layout.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import plotly.graph_objs as go
import math
import warnings
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from models import db, User

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# Flask app configuration
# -----------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# -----------------------------------------------------------
# Disable caching for real-time updates
# -----------------------------------------------------------
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


# -----------------------------------------------------------
# Home (requires login)
# -----------------------------------------------------------
@app.route('/')
@login_required
def index():
    return render_template('index.html')


# -----------------------------------------------------------
# Fetch and clean Yahoo Finance data
# -----------------------------------------------------------
def get_data(quote):
    print(f"📡 Fetching data for {quote}...")
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)

    df = yf.download(quote, start=start, end=end, progress=False)
    if df.empty and not quote.endswith(".NS"):
        print("⚠️ Trying with .NS suffix...")
        df = yf.download(f"{quote}.NS", start=start, end=end, progress=False)
        quote = f"{quote}.NS"

    if df is None or df.empty:
        print(f"❌ No data found for {quote}")
        return None, quote

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column {col} in data for {quote}")
            return None, quote

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"✅ Data fetched successfully for {quote}")
    return df, quote


# -----------------------------------------------------------
# ARIMA Model
# -----------------------------------------------------------
def ARIMA_ALGO(df):
    data = df["Close"].values
    size = int(len(data) * 0.8)
    train, test = data[:size], data[size:]
    history, predictions = list(train), []

    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    rmse = math.sqrt(mean_squared_error(test, predictions))
    arima_pred = float(predictions[-1])

    arima_chart = go.Figure()
    arima_chart.add_trace(go.Scatter(y=test, mode='lines', name='Actual Price', line=dict(color='blue')))
    arima_chart.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted (ARIMA)', line=dict(color='orange')))
    arima_chart.update_layout(title="ARIMA Prediction vs Actual", xaxis_title="Days", yaxis_title="Price", template="plotly_white")

    return arima_pred, rmse, arima_chart.to_html(full_html=False)


# -----------------------------------------------------------
# Linear Regression Model
# -----------------------------------------------------------
def LIN_REG_ALGO(df):
    forecast_out = 7
    df["Close after n days"] = df["Close"].shift(-forecast_out)
    df = df[["Close", "Close after n days"]].dropna()

    X = np.array(df[["Close"]], dtype=float)
    y = np.array(df["Close after n days"], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    forecast = model.predict(X[-forecast_out:])
    lr_pred, mean_forecast = float(forecast[0]), float(forecast.mean())
    rmse = math.sqrt(mean_squared_error(y, model.predict(X)))

    lr_chart = go.Figure()
    lr_chart.add_trace(go.Scatter(y=y[-50:], mode='lines', name='Actual', line=dict(color='blue')))
    lr_chart.add_trace(go.Scatter(y=model.predict(X)[-50:], mode='lines', name='Predicted (LR)', line=dict(color='green')))
    lr_chart.update_layout(title="Linear Regression Prediction", xaxis_title="Days", yaxis_title="Price", template="plotly_white")

    return forecast, lr_pred, mean_forecast, rmse, lr_chart.to_html(full_html=False)


# -----------------------------------------------------------
# Prediction route
# -----------------------------------------------------------
@app.route('/insertintotable', methods=['POST'])
@login_required
def insertintotable():
    nm = request.form['nm'].strip().upper()
    if not nm:
        return render_template('index.html', not_found=True)

    df, nm = get_data(nm)
    if df is None or df.empty:
        return render_template('index.html', not_found=True)

    arima_pred, error_arima, arima_html = ARIMA_ALGO(df)
    forecast, lr_pred, mean_val, error_lr, lr_html = LIN_REG_ALGO(df)

    today_row = df.iloc[-1]
    open_v, high_v, low_v, close_v = today_row["Open"], today_row["High"], today_row["Low"], today_row["Close"]
    adj_close_v, vol_v = today_row["Adj Close"], today_row["Volume"]

    idea = "RISE" if close_v < mean_val else "FALL"
    decision = "BUY" if idea == "RISE" else "SELL"

    return render_template(
        'results.html',
        quote=nm,
        open_s=round(open_v, 2),
        high_s=round(high_v, 2),
        low_s=round(low_v, 2),
        close_s=round(close_v, 2),
        adj_close=round(adj_close_v, 2),
        vol=int(vol_v),
        arima_pred=round(arima_pred, 2),
        lr_pred=round(lr_pred, 2),
        error_arima=round(error_arima, 2),
        error_lr=round(error_lr, 2),
        idea=idea,
        decision=decision,
        arima_chart=arima_html,
        lr_chart=lr_html
    )


# -----------------------------------------------------------
# Extra pages
# -----------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')
@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html')



# -----------------------------------------------------------
# Authentication
# -----------------------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("✅ Account created! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("❌ Incorrect username or password", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


# -----------------------------------------------------------
# Run Flask App
# -----------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()   # creates the users table if not exists
    app.run(debug=True)
