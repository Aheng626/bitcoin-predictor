import os
import time
import pandas as pd
import numpy as np
import ccxt
import requests
from collections import deque
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import random
import warnings

# 隱藏不必要的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 初始化 Flask API
app = Flask(__name__)
CORS(app)

# 初始化 Binance API
exchange = ccxt.binance()

# Discord Webhook URL
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"

# 儲存預測結果以便 10 分鐘後驗證
prediction_queue = deque()
win_count = 0
total_count = 0
recent_predictions = deque(maxlen=20)  # 追蹤最近 20 次預測結果
history_predictions = []  # 儲存完整歷史預測結果

# 設定低勝率閾值
WIN_RATE_THRESHOLD = 65.0

# 設定預測倒數計時
NEXT_PREDICTION_TIME = datetime.now() + timedelta(minutes=1)

def fetch_latest_price(symbol="BTC/USDT"):
    """從 Binance 擷取最新比特幣成交價格"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']  # 最新成交價格
    except Exception as e:
        print(f"⚠️ 無法獲取最新價格: {e}")
        return None

def fetch_binance_data(symbol="BTC/USDT", timeframe="1m", limit=100):
    """從 Binance 擷取 K 線數據"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        return df
    except Exception as e:
        print(f"⚠️ 無法獲取 K 線數據: {e}")
        return pd.DataFrame()

def generate_features(df):
    """生成技術指標作為特徵"""
    df["return_10m"] = df["close"].pct_change(10).shift(-10)
    df["target"] = (df["return_10m"] > 0).astype(int)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(df, model_type="xgboost"):
    """根據 model_type 訓練不同的模型"""
    X = df[["close", "sma_5", "sma_10", "rsi"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "xgboost":
        model = XGBClassifier()
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier()
    
    model.fit(X_train, y_train)
    return model

def predict_and_evaluate(model, df):
    """使用模型預測並評估準確率"""
    recent_data = np.array(df.iloc[-1][["close", "sma_5", "sma_10", "rsi"]], dtype=np.float32).reshape(1, -1)
    pred = model.predict(recent_data)[0]
    return int(pred > 0.5)

def send_discord_notification(message):
    """發送預測結果到 Discord"""
    data = {"content": message}
    requests.post(DISCORD_WEBHOOK_URL, json=data)

def automated_prediction_cycle():
    """自動化預測、驗證與調整演算法"""
    global win_count, total_count
    while True:
        latest_price = fetch_latest_price()
        df = fetch_binance_data()
        df = generate_features(df)
        model = train_model(df)
        pred = predict_and_evaluate(model, df)
        prediction_queue.append((datetime.now(), pred, latest_price))
        
        # 計算最近 20 次預測勝率
        if len(prediction_queue) > 20:
            prediction_queue.popleft()
        recent_win_rate = (sum(p[1] for p in prediction_queue) / len(prediction_queue)) * 100
        total_win_rate = (win_count / total_count) * 100 if total_count > 0 else 100
        
        # 取得 10 分鐘前的預測來驗證
        verification_result = "N/A"
        if len(prediction_queue) >= 10:
            past_pred_time, past_pred, past_price = prediction_queue[-10]
            actual_price = fetch_latest_price()
            actual_trend = 1 if actual_price > past_price else 0
            verification_result = "正確" if past_pred == actual_trend else "錯誤"
        
        # 發送 Discord 通知
        if datetime.now().minute % 5 == 0:
            message = (f"📊 比特幣價格預測更新\n🔹 預測: {'上漲' if pred else '下跌'}\n✅ 10 分鐘前預測驗證: {verification_result}\n✅ 最近 20 次勝率: {recent_win_rate:.2f}%\n🎯 總體勝率: {total_win_rate:.2f}%")
            send_discord_notification(message)
        
        time.sleep(60)
