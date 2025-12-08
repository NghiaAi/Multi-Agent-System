import os
import ta
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from phi.agent import Agent
from phi.model.groq import Groq
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent.parent  
MODELS_DIR = BASE_DIR / "models"    
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

models = {}
for model_path in MODELS_DIR.glob("*_model.joblib"):
    ticker = model_path.stem.replace("_model", "").upper()
    try:
        models[ticker] = joblib.load(model_path)
        print(f"Loaded model for {ticker}")
    except Exception as e:
        print(f"Failed to load {model_path.name}: {e}")

print("Loaded tickers:", list(models.keys()))

def add_enhanced_features(df):
    d = df.copy()
    d['ret_1d'] = d['Adj Close'].pct_change(1)
    d['ret_2d'] = d['Adj Close'].pct_change(2)
    d['ret_5d'] = d['Adj Close'].pct_change(5)
    d['ret_10d'] = d['Adj Close'].pct_change(10)
    d['ret_20d'] = d['Adj Close'].pct_change(20)
    d['overnight_ret'] = d['Open'] / d['Adj Close'].shift(1) - 1
    d['intraday_ret'] = d['Adj Close'] / d['Open'] - 1
    d['hl_spread'] = (d['High'] - d['Low']) / d['Adj Close']
    d['close_position'] = (d['Adj Close'] - d['Low']) / (d['High'] - d['Low'] + 1e-10)
    
    # Momentum
    d['rsi_14'] = ta.momentum.RSIIndicator(d['Adj Close'], window=14).rsi()
    d['rsi_7'] = ta.momentum.RSIIndicator(d['Adj Close'], window=7).rsi()
    macd = ta.trend.MACD(d['Adj Close'])
    d['macd_diff'] = macd.macd_diff()
    d['macd'] = macd.macd()
    d['macd_signal'] = macd.macd_signal()
    stoch = ta.momentum.StochasticOscillator(d['High'], d['Low'], d['Adj Close'])
    d['stoch_k'] = stoch.stoch()
    
    # EMA & crossovers
    ema_12 = d['Adj Close'].ewm(span=12).mean()
    ema_26 = d['Adj Close'].ewm(span=26).mean()
    ema_20 = d['Adj Close'].ewm(span=20).mean()
    ema_50 = d['Adj Close'].ewm(span=50).mean()
    d['ema_cross_12_26'] = (ema_12 > ema_26).astype(int)
    d['ema_cross_20_50'] = (ema_20 > ema_50).astype(int)
    d['ema_cross_change'] = d['ema_cross_12_26'].diff()
    d['macd_cross'] = (d['macd'] > d['macd_signal']).astype(int)
    d['dist_ema_20'] = (d['Adj Close'] - ema_20) / ema_20
    d['dist_ema_50'] = (d['Adj Close'] - ema_50) / ema_50

    # Volatility
    d['volatility_10'] = d['ret_1d'].rolling(10).std()
    d['volatility_20'] = d['ret_1d'].rolling(20).std()
    atr = ta.volatility.AverageTrueRange(d['High'], d['Low'], d['Adj Close'], 14)
    d['atr_ratio'] = atr.average_true_range() / d['Adj Close']
    bb = ta.volatility.BollingerBands(d['Adj Close'], window=20)
    d['bb_position'] = (d['Adj Close'] - bb.bollinger_lband()) / \
                       (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    d['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / d['Adj Close']

    # Trend strength
    adx = ta.trend.ADXIndicator(d['High'], d['Low'], d['Adj Close'], 14)
    d['adx'] = adx.adx()

    # VWAP & Volume
    typical_price = (d['High'] + d['Low'] + d['Adj Close']) / 3
    d['vwap_20'] = (typical_price * d['Volume']).rolling(20).sum() / \
                   d['Volume'].rolling(20).sum()
    d['price_to_vwap'] = d['Adj Close'] / d['vwap_20'] - 1
    d['vol_ratio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    d['vol_momentum'] = d['Volume'] / d['Volume'].shift(5) - 1
    d['mfi'] = ta.volume.MFIIndicator(d['High'], d['Low'], d['Adj Close'], d['Volume'], 14).money_flow_index()

    # Lags & Stats
    d['ret_1d_lag1'] = d['ret_1d'].shift(1)
    d['ret_1d_lag2'] = d['ret_1d'].shift(2)
    d['ret_1d_lag3'] = d['ret_1d'].shift(3)
    d['ret_1d_lag5'] = d['ret_1d'].shift(5)
    d['rsi_14_lag1'] = d['rsi_14'].shift(1)
    d['vol_ratio_lag1'] = d['vol_ratio'].shift(1)
    d['ret_mean_10'] = d['ret_1d'].rolling(10).mean()
    d['ret_zscore_20'] = (d['ret_1d'] - d['ret_1d'].rolling(20).mean()) / \
                         (d['ret_1d'].rolling(20).std() + 1e-10)
    d['ret_skew_20'] = d['ret_1d'].rolling(20).skew()

    # Price patterns
    d['higher_high'] = (d['High'] > d['High'].shift(1)).astype(int)
    d['lower_low'] = (d['Low'] < d['Low'].shift(1)).astype(int)
    d['gap_up'] = (d['Open'] > d['High'].shift(1)).astype(int)
    d['gap_down'] = (d['Open'] < d['Low'].shift(1)).astype(int)
    
    d['vol_regime'] = (d['volatility_10'] > d['volatility_20']).astype(int)
    d['rsi_oversold'] = (d['rsi_14'] < 30).astype(int)
    d['rsi_overbought'] = (d['rsi_14'] > 70).astype(int)
    d['bb_squeeze'] = ((d['bb_position'] < 0.1) | (d['bb_position'] > 0.9)).astype(int)

    # Cumulative & interaction features
    d['cum_ret_5'] = (1 + d['ret_1d']).rolling(5).apply(lambda x: x.prod(), raw=True) - 1
    d['cum_ret_10'] = (1 + d['ret_1d']).rolling(10).apply(lambda x: x.prod(), raw=True) - 1
    d['ret_vol_interaction'] = d['ret_1d'] * d['volatility_10']
    d['rsi_volume_interaction'] = d['rsi_14'] * d['vol_ratio']
    d['momentum_trend'] = d['ret_5d'] * d['adx']
    d['price_vol_interaction'] = d['close_position'] * d['vol_ratio']
    return d

def predict_signal(ticker: str, df: pd.DataFrame) -> dict:
    ticker = ticker.upper()
    if ticker not in models:
        return {"ticker": ticker, "signal": "UNKNOWN", "confidence": 0.0, "reason": "Model not found"}

    model_bundle = models[ticker]
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    df_feat = add_enhanced_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    if len(df_feat) == 0:
        return {"ticker": ticker, "signal": "HOLD", "confidence": 0.0, "reason": "Not enough data"}

    X = df_feat[features].iloc[[-1]]
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[0]
    idx = np.argmax(probs)
    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    signal = signal_map[idx]
    confidence = float(probs[idx])

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 3),
        "probabilities": {signal_map[i]: float(p) for i, p in enumerate(probs)},
        "feature_values": X.iloc[0].to_dict()  
    }


def create_genai_explainer(features: list):
    feature_list_str = ", ".join(features)
    system_prompt = f"""
        You are a financial AI analyst assistant.
        Given a model decision output (BUY, HOLD, or SELL), confidence score, and the most recent values of all features,
        explain the reasoning behind the recommendation in a short, professional, and data-driven style.
        Use the actual numerical values of features in your explanation where relevant.
        Format the answer as JSON:
        {{"signal": "...", "confidence": ..., "explanation": "..."}}
        """
    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=30,
            temperature=0.2,
            max_tokens=1000,
        ),
        system_prompt=system_prompt,
        debug_mode=True,
    )

def run_trading_agent(query: str, sql_result: list = None, ticker: str = None) -> dict:
    if not sql_result or not isinstance(sql_result, list) or len(sql_result) == 0:
        return {"status": "error", "message": "No data for trading decision."}

    if not ticker or ticker.strip() == "":
        return {"status": "error", "message": "Ticker not provided or invalid."}

    df = pd.DataFrame(sql_result)

    decision = predict_signal(ticker, df)

    features = list(decision["feature_values"].keys())
    explainer = create_genai_explainer(features)

    expl_input = json.dumps({
        "decision": decision,
        "feature_values": decision["feature_values"]
    })
    explanation = explainer.run(expl_input)

    try:
        explanation_json = json.loads(explanation.content)
    except Exception:
        explanation_json = {"explanation": explanation.content.strip()}

    result = {**decision, **explanation_json}
    return {"status": "success", "decision": result}
