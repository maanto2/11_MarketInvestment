# market_dashboard.py
"""
This script runs the Dash dashboard, reading data from the researcher's REST API.
No market data fetching or heavy computation is performed here.
"""
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import threading
import time
from datetime import datetime
import pytz
import requests
import pandas as pd

# Data store for S&P 500
sp500_data_store = {
    "actual": [],
    "predicted": [],
    "timestamps": [],
    "sentiment_score": []
}
# Data store for NIFTY 50
nifty_data_store = {
    "actual": [],
    "predicted": [],
    "timestamps": [],
    "sentiment_score": []
}

RESEARCHER_API_URL = "http://192.168.1.112:5000/api"  # <-- Set your Pi's IP here

def fetch_raw_market_data():
    # Fetch S&P 500 and NIFTY raw data from Pi
    try:
        sp500_resp = requests.get(f"{RESEARCHER_API_URL}/sp500_raw", timeout=10)
        if sp500_resp.status_code == 200:
            sp500_hist = pd.read_json(sp500_resp.text)
        else:
            sp500_hist = pd.DataFrame()
    except Exception as e:
        print(f"Error fetching S&P 500 raw data: {e}")
        sp500_hist = pd.DataFrame()
    try:
        nifty_resp = requests.get(f"{RESEARCHER_API_URL}/nifty_raw", timeout=10)
        if nifty_resp.status_code == 200:
            nifty_hist = pd.read_json(nifty_resp.text)
        else:
            nifty_hist = pd.DataFrame()
    except Exception as e:
        print(f"Error fetching NIFTY raw data: {e}")
        nifty_hist = pd.DataFrame()
    return sp500_hist, nifty_hist

def fetch_macro_data():
    try:
        macro_resp = requests.get(f"{RESEARCHER_API_URL}/macro", timeout=10)
        if macro_resp.status_code == 200:
            return macro_resp.json()
    except Exception as e:
        print(f"Error fetching macro data: {e}")
    return {}

# In periodic_reload, fetch raw data and run ML/sentiment locally
from marketPrediction4 import (
    train_lstm_model, predict_lstm, fetch_sentiment_finbert, fetch_sentiment_india_news
)

# Only load the last 7 days of data from CSV for plotting
try:
    df = pd.read_csv('dashboard_data.csv', parse_dates=['timestamp'])
    one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    df = df[df['timestamp'] >= one_week_ago]
    sp500_data_store['timestamps'] = list(df['timestamp'])
    sp500_data_store['actual'] = list(df['actual'])
    sp500_data_store['predicted'] = list(df['predicted'])
    sp500_data_store['sentiment_score'] = list(df['sentiment_score'])
except Exception:
    pass
try:
    df = pd.read_csv('dashboard_data_nifty.csv', parse_dates=['timestamp'])
    one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    df = df[df['timestamp'] >= one_week_ago]
    nifty_data_store['timestamps'] = list(df['timestamp'])
    nifty_data_store['actual'] = list(df['actual'])
    nifty_data_store['predicted'] = list(df['predicted'])
    nifty_data_store['sentiment_score'] = list(df['sentiment_score'])
except Exception:
    pass

def periodic_reload():
    while True:
        sp500_hist, nifty_hist = fetch_raw_market_data()
        # S&P 500 ML and sentiment
        if not sp500_hist.empty:
            try:
                model_lstm, scaler_lstm = train_lstm_model(sp500_hist)
                prediction = predict_lstm(model_lstm, scaler_lstm, sp500_hist)
                sentiment = fetch_sentiment_finbert()
                sp500_data_store['timestamps'] = list(sp500_hist.index)
                sp500_data_store['actual'] = list(sp500_hist['Close'])
                sp500_data_store['predicted'] = [None]*(len(sp500_hist)-1) + [prediction]
                sp500_data_store['sentiment_score'] = [sentiment]*len(sp500_hist)
            except Exception as e:
                print(f"Error processing S&P 500 ML: {e}")
        # NIFTY ML and sentiment
        if not nifty_hist.empty:
            try:
                model_lstm, scaler_lstm = train_lstm_model(nifty_hist)
                prediction = predict_lstm(model_lstm, scaler_lstm, nifty_hist)
                sentiment = fetch_sentiment_india_news()
                nifty_data_store['timestamps'] = list(nifty_hist.index)
                nifty_data_store['actual'] = list(nifty_hist['Close'])
                nifty_data_store['predicted'] = [None]*(len(nifty_hist)-1) + [prediction]
                nifty_data_store['sentiment_score'] = [sentiment]*len(nifty_hist)
            except Exception as e:
                print(f"Error processing NIFTY ML: {e}")
        time.sleep(30)  # Reload every 30 seconds

# Start background thread to reload data
threading.Thread(target=periodic_reload, daemon=True).start()

app = Dash(__name__)
app.title = "S&P 500 & NIFTY 50 Prediction Dashboard (Read-Only)"

app.layout = html.Div([
    html.H1("Live S&P 500 & NIFTY 50 Prediction Dashboard (Read-Only)"),
    html.Div([
        html.Div([
            html.H2("S&P 500"),
            dcc.Graph(id='trend-graph'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H2("NIFTY 50 (India)"),
            dcc.Graph(id='nifty-trend-graph'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    dcc.Interval(id='interval', interval=60*1000, n_intervals=0)
])

def detect_sudden_changes(prices, timestamps, sentiment_scores=None):
    annotations = []
    threshold = 0.02  # 2% change
    for i in range(1, len(prices)):
        prev = prices[i-1]
        curr = prices[i]
        if prev is None or curr is None:
            continue
        change = (curr - prev) / prev if prev != 0 else 0
        if abs(change) >= threshold:
            sentiment = None
            if sentiment_scores and i < len(sentiment_scores):
                sentiment = sentiment_scores[i]
            reason = f"{'Spike' if change > 0 else 'Drop'}: {change*100:.2f}%"
            if sentiment is not None:
                if sentiment > 0.5:
                    reason += f"\nPositive sentiment ({sentiment:.2f})"
                elif sentiment < -0.5:
                    reason += f"\nNegative sentiment ({sentiment:.2f})"
            annotations.append(dict(
                x=timestamps[i],
                y=curr,
                xref='x',
                yref='y',
                text=reason,
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40 if change > 0 else 40,
                bgcolor='rgba(255,255,0,0.7)' if change > 0 else 'rgba(255,0,0,0.5)',
                bordercolor='black',
                font=dict(size=12)
            ))
    return annotations

@app.callback(
    Output('trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    if len(sp500_data_store["timestamps"]) == 0:
        return go.Figure()
    window = 5
    predicted_smoothed = pd.Series(sp500_data_store["predicted"]).rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp500_data_store["timestamps"], y=sp500_data_store["actual"],
                             mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=sp500_data_store["timestamps"], y=predicted_smoothed,
                             mode='lines+markers', name='Predicted (Smoothed)'))
    # Add annotations for sudden changes
    annotations = detect_sudden_changes(
        sp500_data_store["actual"],
        sp500_data_store["timestamps"],
        sp500_data_store.get("sentiment_score")
    )
    fig.update_layout(
        title='S&P 500 Actual vs Predicted',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        template='plotly_dark',
        annotations=annotations
    )
    return fig

@app.callback(
    Output('nifty-trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_nifty_graph(n):
    if len(nifty_data_store["timestamps"]) == 0:
        return go.Figure()
    window = 5
    predicted_smoothed = pd.Series(nifty_data_store["predicted"]).rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nifty_data_store["timestamps"], y=nifty_data_store["actual"],
                             mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=nifty_data_store["timestamps"], y=predicted_smoothed,
                             mode='lines+markers', name='Predicted (Smoothed)'))
    # Add annotations for sudden changes
    annotations = detect_sudden_changes(
        nifty_data_store["actual"],
        nifty_data_store["timestamps"],
        nifty_data_store.get("sentiment_score")
    )
    fig.update_layout(
        title='NIFTY 50 Actual vs Predicted',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        template='plotly_dark',
        annotations=annotations
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)

