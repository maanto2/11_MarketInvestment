# market_researcher.py
"""
This script fetches all market data and macroeconomic indicators for the S&P 500 and NIFTY 50.
It writes results to CSV files for the dashboard to consume. Intended to run on a remote Raspberry Pi or similar device.
"""
import os
import time
import threading
import logging
import psutil
from flask import Flask, jsonify
# Only import raw data fetchers and macro fetchers from marketPrediction4
from marketPrediction4 import (
    get_sp500_data_for_live, get_nifty_data_for_live,
    fetch_sector_performance_data, fetch_earnings_season_data,
    fetch_gdp_data, fetch_inflation_data, fetch_fed_policy_data,
    fetch_global_events_data, fetch_government_policy_data,
    fetch_bond_yields_data, fetch_commodities_data, fetch_currency_data,
    fetch_volatility_and_global_features
)

app = Flask(__name__)

@app.route('/api/sp500_raw', methods=['GET'])
def get_sp500_raw():
    hist = get_sp500_data_for_live()
    return hist.tail(100).to_json(date_format='iso')

@app.route('/api/nifty_raw', methods=['GET'])
def get_nifty_raw():
    hist = get_nifty_data_for_live()
    return hist.tail(100).to_json(date_format='iso')

@app.route('/api/macro', methods=['GET'])
def get_macro():
    # Fetch all macro data in parallel (as before)
    import concurrent.futures
    fetchers = {
        'sector_performance': fetch_sector_performance_data,
        'earnings_season': fetch_earnings_season_data,
        'gdp': fetch_gdp_data,
        'inflation': fetch_inflation_data,
        'fed_policy': fetch_fed_policy_data,
        'global_events': fetch_global_events_data,
        'government_policy': fetch_government_policy_data,
        'bond_yields': fetch_bond_yields_data,
        'commodities': fetch_commodities_data,
        'currency': fetch_currency_data,
        'volatility_global': fetch_volatility_and_global_features,
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in fetchers.items()}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = None
    return jsonify(results)

def log_system_stats():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        logging.info(f"[SYSTEM] CPU: {cpu:.1f}% | RAM: {ram:.1f}%")
        time.sleep(30)

def start_update_threads():
    threading.Thread(target=log_system_stats, daemon=True).start()

def profile_fetcher(fetcher, *args, **kwargs):
    import tracemalloc
    import time
    tracemalloc.start()
    start_time = time.perf_counter()
    result = None
    try:
        result = fetcher(*args, **kwargs)
    except Exception as e:
        print(f"[PROFILE ERROR] {fetcher.__name__}: {e}")
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[PROFILE] {fetcher.__name__}: {elapsed:.2f}s, {peak/1024/1024:.2f} MiB peak memory")
    return result

# Example usage: profile all fetchers at startup
if __name__ == "__main__":
    fetchers = [
        get_sp500_data_for_live,
        get_nifty_data_for_live,
        fetch_sector_performance_data,
        fetch_earnings_season_data,
        fetch_gdp_data,
        fetch_inflation_data,
        fetch_fed_policy_data,
        fetch_global_events_data,
        fetch_government_policy_data,
        fetch_bond_yields_data,
        fetch_commodities_data,
        fetch_currency_data,
        fetch_volatility_and_global_features,
    ]
    for fn in fetchers:
        profile_fetcher(fn)
    start_update_threads()
    app.run(host='0.0.0.0', port=5000, debug=False)
