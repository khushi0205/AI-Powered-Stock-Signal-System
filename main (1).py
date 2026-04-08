from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import yfinance as yf
import requests
import os
import uuid
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import json
import threading

load_dotenv()

app = FastAPI(title="AI Financial Intelligence API")

# ================================

# 🔥 CONFIG

# ================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,TSLA").split(",")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================================

# 🔥 GLOBAL CACHE

# ================================

analysis_cache: Dict[str, Dict] = {}

# ================================

# 🔥 COMPANY NAMES

# ================================

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia"
}

# ================================

# 🔥 MODELS

# ================================

class StocksRequest(BaseModel):
    tickers: List[str]

# ================================

# 🔥 CORE FUNCTIONS (FAST + RELIABLE)

# ================================
def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    try:
        stock = yf.download(ticker, period=period, interval=interval, progress=False)
        return stock.tail(30)
    except:
        return pd.DataFrame()


def calculate_percentage_change(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return 0.0

    try:
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0.0


def calculate_volatility(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return "Low"

    try:
        returns = df["Close"].pct_change().dropna()
        vol = returns.std()

        if vol > 0.02:
            return "High"
        elif vol > 0.01:
            return "Moderate"
        else:
            return "Low"
    except:
        return "Low"


def get_historical_trend(ticker: str):
    try:
        stock = yf.Ticker(ticker)

        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 50:
            return {
                "3m_change": 0,
                "6m_change": 0,
                "12m_change": 0,
                "classification": "Neutral",
                "volatility": "Low"
            }

        close = hist["Close"]

        # 🔥 slices instead of multiple API calls
        c3 = ((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]) * 100
        c6 = ((close.iloc[-1] - close.iloc[-120]) / close.iloc[-120]) * 100
        c12 = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100

        returns = close.pct_change().dropna()
        vol = returns.std()

        if vol > 0.02:
            volatility = "High"
        elif vol > 0.01:
            volatility = "Moderate"
        else:
            volatility = "Low"

        if c3 > 5:
            classification = "Bullish"
        elif c3 < -5:
            classification = "Bearish"
        else:
            classification = "Neutral"

        return {
            "3m_change": round(float(c3), 2),
            "6m_change": round(float(c6), 2),
            "12m_change": round(float(c12), 2),
            "classification": classification,
            "volatility": volatility
        }

    except Exception as e:
        print("❌ Trend error:", e)
        return {
            "3m_change": 0,
            "6m_change": 0,
            "12m_change": 0,
            "classification": "Neutral",
            "volatility": "Low"
        }
def get_stock_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty:
            hist = stock.history(period="1mo")

        if hist.empty:
            return 0.0

        return float(hist["Close"].iloc[-1])
    except:
        return 0.0

def get_news(ticker: str, limit: int = 3) -> List[Dict]:
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&lang=en-US&region=US&quotesCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10).json()

        headlines = []

        for news in response.get("news", [])[:limit]:
            title = news.get("title", "")
            link = news.get("link", "")

            if not title or not link:
                continue

            # 🔥 NO scraping (avoids crashes)
            headlines.append({
                "title": title,
                "url": link,
                "content": ""
            })

        return headlines

    except Exception as e:
        print("❌ News error:", e)
        return []

def call_llm(ticker, price, news, c3m, c6m, c12m, volatility, classification):
    try:
        prompt = f"""
You are a financial analyst.

Analyze stock: {ticker}

PRICE: {price}

HISTORICAL PERFORMANCE:
- 3M Change: {c3m}%
- 6M Change: {c6m}%
- 12M Change: {c12m}%
- Trend: {classification}
- Volatility: {volatility}

NEWS:
{news}

DECISION RULES:
- Strong bullish (>15% 6M) → BUY
- Moderate bullish (5–15%) → BUY/HOLD
- Weak (0–5%) → HOLD
- Negative → SELL/HOLD
- News can adjust signal by ONE step only

Return ONLY JSON:

{{
  "signal": "BUY or SELL or HOLD",
  "confidence": number between 0 and 1,
  "reasoning": "short explanation using numbers + news"
}}
"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )

        parsed = json.loads(response.choices[0].message.content)

        return {
            "signal": parsed.get("signal", "HOLD"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": parsed.get("reasoning", "")
        }

    except Exception as e:
        return {
            "signal": "HOLD",
            "confidence": 0.3,
            "reasoning": f"Fallback: {str(e)}"
        }

def analyze_ticker(ticker: str):
    price = get_stock_price(ticker)

    # 🔥 NEW: historical data
    trend = get_historical_trend(ticker)

    # 🔥 NEWS
    news_data = get_news(ticker)
    news_titles = [n["title"] for n in news_data]

    llm_result = call_llm(
        ticker,
        price,
        news_titles,
        trend["3m_change"],
        trend["6m_change"],
        trend["12m_change"],
        trend["volatility"],
        trend["classification"]
    )

    return {
        "run_id": str(uuid.uuid4()),
        "ticker": ticker,
        "price": price,

        # 🔥 NEW FIELDS
        "3m_change": trend["3m_change"],
        "6m_change": trend["6m_change"],
        "12m_change": trend["12m_change"],
        "volatility": trend["volatility"],
        "trend_classification": trend["classification"],

        "signal": llm_result["signal"],
        "confidence": llm_result["confidence"],
        "reasoning": llm_result["reasoning"],
        "news": news_data,
        "timestamp": datetime.utcnow().isoformat()
    }

# ================================

# 🔥 BACKGROUND JOB

# ================================

def run_batch_analysis():
    print("🔄 Running batch analysis...")

    for ticker in TICKERS:
        ticker = ticker.strip().upper()

        try:
            print(f"➡️ Processing {ticker}")

            result = analyze_ticker(ticker)

            analysis_cache[ticker] = result
            print(f"✅ Cached {ticker}")

        except Exception as e:
            print(f"❌ Failed {ticker}: {e}")

            # 🔥 STILL store fallback (VERY IMPORTANT)
            analysis_cache[ticker] = {
                "ticker": ticker,
                "signal": "HOLD",
                "confidence": 0.3,
                "reasoning": f"Fallback due to error: {str(e)}",
                "news": [],
                "timestamp": datetime.utcnow().isoformat()
            }

    print("🚀 Cache update complete")


# ================================

# 🔥 STARTUP

# ================================

@app.on_event("startup")
def startup():
    # Run batch in background (non-blocking)
    threading.Thread(target=run_batch_analysis).start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(run_batch_analysis, "interval", minutes=15)
    scheduler.start()


# ================================

# 🔥 FAST ENDPOINT (NO TIMEOUT)

# ================================

@app.post("/analyze/stocks")
async def analyze_stocks(req: StocksRequest):

    results = []
    
    for ticker in req.tickers:
        ticker = ticker.upper()
    
        if ticker in analysis_cache:
            results.append(analysis_cache[ticker])
        else:
            results.append({
                "ticker": ticker,
                "error": "Analysis not ready yet"
            })
    
    return results


# ================================

# 🔥 ADMIN

# ================================

@app.post("/admin/run-now")
def run_now():
    run_batch_analysis()
    return {"status": "cache refreshed"}

@app.get("/cache/status")
def cache_status():
    return {
    "cached_tickers": list(analysis_cache.keys()),
    "count": len(analysis_cache)
    }

# ================================

# 🔥 HEALTH + ROOT

# ================================

@app.get("/")
def root():
    return {
    "project": "AI Financial Intelligence API",
    "mode": "cached-fast",
    "status": "running"
    }

@app.get("/health")
def health():
    return {
    "status": "healthy",
    "cache_size": len(analysis_cache)
    }
