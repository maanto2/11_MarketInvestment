# market_researcher.py
"""
This script fetches all market data, runs models, computes predictions, and collects sentiment for S&P 500 and NIFTY 50.
It writes results to CSV files for the dashboard to consume. Intended to run on a remote Raspberry Pi or similar device.
"""
import os
import time
import threading
from marketPrediction4 import (
    update_data, update_nifty_data, data_store, nifty_data_store
)
from flask import Flask, jsonify

app = Flask(__name__)

def run_researcher():
    # Start both update threads
    threading.Thread(target=update_data, daemon=True).start()
    threading.Thread(target=update_nifty_data, daemon=True).start()
    # Keep alive
    while True:
        time.sleep(60)

@app.route('/api/sp500', methods=['GET'])
def get_sp500():
    # Return the latest S&P 500 data (last 100 points)
    return jsonify({
        'timestamps': data_store['timestamps'][-100:],
        'actual': data_store['actual'][-100:],
        'predicted': data_store['predicted'][-100:],
        'sentiment_score': data_store['sentiment_score'][-100:]
    })

@app.route('/api/nifty', methods=['GET'])
def get_nifty():
    # Return the latest NIFTY 50 data (last 100 points)
    return jsonify({
        'timestamps': nifty_data_store['timestamps'][-100:],
        'actual': nifty_data_store['actual'][-100:],
        'predicted': nifty_data_store['predicted'][-100:],
        'sentiment_score': nifty_data_store['sentiment_score'][-100:]
    })

def start_update_threads():
    threading.Thread(target=update_data, daemon=True).start()
    threading.Thread(target=update_nifty_data, daemon=True).start()

if __name__ == "__main__":
    start_update_threads()
    app.run(host='0.0.0.0', port=5000, debug=False)
