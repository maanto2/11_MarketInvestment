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

def fetch_api_data(endpoint):
    try:
        r = requests.get(f"{RESEARCHER_API_URL}/{endpoint}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Convert timestamps to pandas datetime if needed
            if 'timestamps' in data:
                data['timestamps'] = pd.to_datetime(data['timestamps'])
            return data
    except Exception as e:
        print(f"Error fetching {endpoint} from researcher API: {e}")
    return None

def periodic_reload():
    while True:
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
    fig.update_layout(
        title='S&P 500 Actual vs Predicted',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        template='plotly_dark'
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
    fig.update_layout(
        title='NIFTY 50 Actual vs Predicted',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        template='plotly_dark'
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
