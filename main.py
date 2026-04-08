# Copyright (c) 2025 Khushi Sali. All rights reserved.
#
# PROPRIETARY LICENSE
#
# This software and associated documentation files ("Software") are the 
# confidential and proprietary information of the copyright holder.
#
# Unauthorized copying, modification, distribution, or use of the Software,
# in whole or in part, without express written permission from the copyright 
# holder is strictly prohibited.
#
# The Software is provided "AS IS", without warranty of any kind, express 
# or implied, including but not limited to the warranties of merchantability, 
# fitness for a particular purpose, and non-infringement. In no event shall 
# the copyright holder be liable for any claim, damages, or other liability, 
# whether in an action of contract, tort, or otherwise, arising from, out of, 
# or in connection with the Software or its use.

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Body, Depends, status
from pydantic import BaseModel
import yfinance as yf
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import ccxt
from datetime import datetime, timedelta
import requests
import os 
import uuid
import json
import re
import logging
from typing import List, Optional, Dict, Any
from dateutil.relativedelta import relativedelta
from openai import OpenAI
import asyncio
from fastapi.middleware.cors import CORSMiddleware


from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# load_dotenv()
# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.warning("OpenAI API key not found in environment variables")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
if not CRYPTOPANIC_API_KEY:
    logging.warning("CryptoPanic API key not found in environment variables")

app = FastAPI(
    title="AI Financial News & Impact API",
    description="""
    An AI-powered API for analyzing financial news and predicting market impact.
    
    ## Features
    
    - 📈 Stock market analysis with sentiment scoring
    - ₿ Cryptocurrency analysis with technical indicators
    - 📰 News aggregation from multiple sources
    - 🤖 AI-powered recommendations using FinBERT and GPT models
    - 📊 Performance tracking and backtesting
    
    ## Endpoints
    
    - `/analyze/stocks` - Analyze stock tickers
    - `/analyze/crypto` - Analyze cryptocurrency tickers  
    - `/analyze/mixed` - Analyze mixed asset types
    - `/health` - API health status
    - `/performance/stats` - Performance statistics
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "contact.khushis@gmail.com",
    },
    openapi_tags=[
        {
            "name": "Analysis",
            "description": "Financial asset analysis endpoints",
        },
        {
            "name": "Health", 
            "description": "API health check endpoints",
        },
        {
            "name": "Performance",
            "description": "Performance tracking and statistics",
        }
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global models
finbert_model = None
finbert_tokenizer = None

# ----------------------------
# Load FinBERT at startup
# ----------------------------

def init_log_file():
    try:
        with open("analysis_log.json", "x") as f:
            f.write("")
    except FileExistsError:
        pass


        
# main.py - Update startup_events function
@app.on_event("startup")
def startup_events():
    """Run all startup events"""
    if os.getenv("DISABLE_FINBERT", "false").lower() == "true":
        logging.warning("⚠️ FinBERT disabled via environment variable. Skipping model load.")
    else:
        load_finbert_model()

    validate_api_keys()
    start_scheduler()


def load_finbert_model():
    global finbert_model, finbert_tokenizer

    if os.getenv("DISABLE_FINBERT", "false").lower() == "true":
        logging.warning("⚠️ FinBERT model loading disabled via environment variable.")
        finbert_model = None
        finbert_tokenizer = None
        return

    try:
        init_log_file()
        finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        finbert_model.eval()
        logging.info("✅ FinBERT model loaded and cached at startup.")
    except Exception as e:
        logging.error(f"❌ Error loading FinBERT model: {e}")
        finbert_model = None
        finbert_tokenizer = None


def validate_api_keys():
    """Validate that required API keys are available"""
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OpenAI API key not set. LLM features will be disabled.")
    
    if not os.getenv("CRYPTOPANIC_API_KEY"):
        logging.warning("CryptoPanic API key not set. Crypto news features will be limited.")

# ----------------------------
# Utility Functions
# ----------------------------
def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Get stock data from Yahoo Finance"""
    try:
        stock = yf.download(ticker, period=period, interval=interval, progress=False)
        return stock.tail(30)
    except Exception as e:
        print(f"Error getting stock data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_percentage_change(df: pd.DataFrame) -> float:
    """Calculate percentage change from first to last close price"""
    if df is None or df.empty or len(df) < 2:
        return 0.0
    
    # Handle yfinance column format (tuples)
    close_col = None
    for col in df.columns:
        if isinstance(col, tuple):
            if 'Close' in col:
                close_col = col
                break
        elif col == 'Close':
            close_col = col
            break
    
    if close_col is None or close_col not in df.columns:
        return 0.0
        
    try:
        start_price = df[close_col].iloc[0]
        end_price = df[close_col].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0.0

def calculate_volatility(df: pd.DataFrame) -> str:
    """Calculate volatility based on standard deviation of returns"""
    if df is None or df.empty or len(df) < 2:
        return "Low"
    
    # Handle yfinance column format
    close_col = None
    for col in df.columns:
        if isinstance(col, tuple):
            if 'Close' in col:
                close_col = col
                break
        elif col == 'Close':
            close_col = col
            break
    
    if close_col is None:
        return "Low"
        
    try:
        returns = df[close_col].pct_change().dropna()
        if len(returns) < 2:
            return "Low"
            
        vol_std = returns.std()
        return "High" if vol_std > 0.02 else "Moderate" if vol_std > 0.01 else "Low"
    except:
        return "Low"

def get_historical_trend(ticker: str, is_crypto: bool = False, exchange: str = "yahoo") -> Dict[str, Any]:
    """Get 3-month, 6-month, and 12-month trend"""
    try:
        end_date = datetime.now()
        start_date_3m = end_date - timedelta(days=90)
        start_date_6m = end_date - timedelta(days=180)
        start_date_12m = end_date - timedelta(days=365)

        # Format ticker for yfinance
        if is_crypto:
            yf_ticker = ticker.replace("/", "-") if "/" in ticker else f"{ticker}-USD"
        else:
            yf_ticker = ticker

        # Download data
        df_3m = yf.download(yf_ticker, start=start_date_3m, end=end_date, progress=False)
        df_6m = yf.download(yf_ticker, start=start_date_6m, end=end_date, progress=False)
        df_12m = yf.download(yf_ticker, start=start_date_12m, end=end_date, progress=False)

        if df_3m.empty or df_6m.empty or df_12m.empty:
            return {"error": "No data available", "3m_change": 0, "6m_change": 0, "12m_change": 0, "classification": "Neutral", "volatility": "Low"}

        # Calculate changes
        change_3m = calculate_percentage_change(df_3m)
        change_6m = calculate_percentage_change(df_6m)
        change_12m = calculate_percentage_change(df_12m)

        # Classification
        if change_3m > 5:
            classification = "Bullish"
        elif change_3m < -5:
            classification = "Bearish"
        else:
            classification = "Neutral"

        # Volatility
        volatility = calculate_volatility(df_3m)

        return {
            "3m_change": float(round(change_3m, 2)),
            "6m_change": float(round(change_6m, 2)),
            "12m_change": float(round(change_12m, 2)),
            "classification": classification,
            "volatility": volatility
        }

    except Exception as e:
        return {"error": str(e), "3m_change": 0, "6m_change": 0, "12m_change": 0, "classification": "Neutral", "volatility": "Low"}

def get_crypto_ohlcv(symbol="BTC/USDT", exchange="binance", days=90):
    try:
        ex = getattr(ccxt, exchange)()
        ex.load_markets()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", limit=days)
    except Exception as e:
        print(f"[ccxt] Error fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def get_crypto_historical_trend(symbol="ETH/USDT", exchange="binance"):
    """Get proper 3m, 6m, 12m trends for crypto"""
    try:
        # Get 12 months data
        df_12m = get_crypto_ohlcv(symbol, exchange, days=365)
        df_6m = df_12m.tail(180) if len(df_12m) >= 180 else df_12m
        df_3m = df_12m.tail(90) if len(df_12m) >= 90 else df_12m
        
        if df_3m.empty:
            return {"3m_change": 0, "6m_change": 0, "12m_change": 0, "classification": "Neutral", "volatility": "Low"}
        
        # Calculate changes
        change_3m = ((df_3m["close"].iloc[-1] - df_3m["close"].iloc[0]) / df_3m["close"].iloc[0]) * 100
        change_6m = ((df_6m["close"].iloc[-1] - df_6m["close"].iloc[0]) / df_6m["close"].iloc[0]) * 100
        change_12m = ((df_12m["close"].iloc[-1] - df_12m["close"].iloc[0]) / df_12m["close"].iloc[0]) * 100
        
        # Classification
        if change_3m > 10:  # Higher threshold for crypto
            classification = "Bullish"
        elif change_3m < -10:
            classification = "Bearish"
        else:
            classification = "Neutral"
            
        # Volatility
        daily_returns = df_3m["close"].pct_change().dropna()
        vol_std = daily_returns.std()
        volatility = "High" if vol_std > 0.03 else "Moderate" if vol_std > 0.015 else "Low"
        
        return {
            "3m_change": round(float(change_3m), 2),
            "6m_change": round(float(change_6m), 2), 
            "12m_change": round(float(change_12m), 2),
            "classification": classification,
            "volatility": volatility
        }
        
    except Exception as e:
        return {"error": str(e), "3m_change": 0, "6m_change": 0, "12m_change": 0, "classification": "Neutral", "volatility": "Low"}

def get_crypto_news(symbol: str, limit: int = 10) -> List[Dict]:
    """Get proper crypto news with real URLs and content"""
    if not CRYPTOPANIC_API_KEY:
        logging.warning("CryptoPanic API key not available")
        return []
    
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies={symbol}&kind=news"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        news_items = []
        for post in data.get('results', [])[:limit]:
            news_items.append({
                "title": post.get('title', ''),
                "url": post.get('url', ''),
                "content": post.get('title', ''),  # CryptoPanic only gives titles
                "published_at": post.get('published_at', '')
            })
        return news_items
    except Exception as e:
        print(f"Error fetching crypto news for {symbol}: {e}")
        # Fallback to empty list instead of breaking
        return []
    
def fetch_article_text(url: str) -> str:
    """Fetch article text from URL"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs[:3]) if paragraphs else ""
    except Exception as e:
        print(f"Error fetching article {url}: {e}")
        return ""

COMPANY_NAMES = {
    "NVDA": "NVIDIA", "MSFT": "Microsoft", "AAPL": "Apple", 
    "GOOGL": "Alphabet", "AMZN": "Amazon", "TSLA": "Tesla",
    "WBA": "Walgreens", "MMM": "3M", "EL": "Estee Lauder",
    "PYPL": "PayPal", "T": "AT&T", "IBM": "IBM", 
    "WMT": "Walmart", "VZ": "Verizon"
}

def get_news_headlines(ticker: str, limit: int = 10) -> List[Dict]:
    """Get news headlines for a stock"""
    company_name = COMPANY_NAMES.get(ticker.upper(), ticker.upper())
    
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&lang=en-US&region-{limit}&quotesCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15).json()
        
        headlines = []
        for news in response.get("news", [])[:limit]:
            title = news.get("title", "")
            link = news.get("link", "")
            
            if not title or not link:
                continue

            article_text = fetch_article_text(link)
            
            # Check if relevant to the company
            if (ticker.upper() in title.upper() or 
                company_name.upper() in title.upper() or
                ticker.upper() in article_text.upper() or 
                company_name.upper() in article_text.upper()):
                
                headlines.append({
                    "title": title,
                    "url": link,
                    "content": article_text
                })
                
        return headlines
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def get_finbert_sentiment(text: str) -> float:
    """Better sentiment for short crypto news"""
    if not text or len(text.strip()) < 10:  # Skip very short text
        return 0.0
        
    if finbert_model is None:
        return 0.0
        
    try:
        # Preprocess crypto-specific text
        text = text.replace("$", "").replace("#", "")  # Remove crypto symbols
        
        inputs = finbert_tokenizer(
            text[:256],  # Shorter for crypto news
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][2].item() - probs[0][0].item()  # positive - negative
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0.0

def classify_relevance(title: str, is_crypto: bool = False) -> str:
    """Classify headline relevance"""
    title_lower = title.lower()
    
    # Enhanced relevance signals
    high_signals = [
        r"\bearnings", r"\bguidance", r"\bforecast", r"\bdowngrade", r"\bupgrade",
        r"\bacquisition", r"\bmerger", r"\blawsuit", r"\bsec", r"\bfda", r"\bipo",
        r"\blayoff", r"\bdividend", r"\bbuyback", r"\bceo", r"\bcfo", r"\bexecutive",
        r"\binsider", r"\binvestigation", r"\bsettlement", r"\bfine", r"\bpenalty",
        r"\bprofit", r"\bloss", r"\brevenue", r"\bmargin", r"\bgrowth"
    ]
    
    medium_signals = [
        r"\banalyst", r"\bprice target", r"\brating", r"\binitiate", r"\bcover",
        r"\bmarket", r"\beconomy", r"\binflation", r"\brates", r"\bfed",
        r"\bproduct", r"\blaunch", r"\bupdate", r"\bpartnership", r"\bdeal",
        r"\bcontract", r"\bexpansion", r"\bstrategy", r"\boutlook"
    ]

    if any(re.search(sig, title_lower) for sig in high_signals):
        return "High"
    elif any(re.search(sig, title_lower) for sig in medium_signals):
        return "Medium"
    else:
        return "Low"

def score_event(sentiment: float, price_trend: str, hist_class: str = "Neutral", is_crypto: bool = False) -> tuple:
    """Compute impact score"""
    score = 0

    # Sentiment weighting
    if is_crypto:
        if sentiment > 0.2: score += 1
        elif sentiment < -0.2: score -= 1
    else:
        if sentiment > 0.2: score += 0.3
        elif sentiment < -0.2: score -= 0.3

    # Short-term trend
    if price_trend == "↑": score += 2
    elif price_trend == "↓": score -= 2

    # Historical classification
    if is_crypto:
        if hist_class == "Bullish": score += 3
        elif hist_class == "Bearish": score -= 3
    else:
        if hist_class == "Bullish": score += 4
        elif hist_class == "Bearish": score -= 4

    score += sentiment * 0.5
    
    # Final classification
    if score >= 2:
        return "Bullish (+)", round(score, 3)
    elif score <= -2:
        return "Bearish (-)", round(score, 3)
    else:
        return "Neutral", round(score, 3)

def get_momentum_strength(trend_3m: float, trend_6m: float) -> str:
    """Realistic momentum strength classification"""
    if trend_6m > 20: return "Strong Bullish"
    elif trend_6m > 10: return "Moderate Bullish"
    elif trend_6m > 0: return "Weak Bullish"
    elif trend_6m > -10: return "Weak Bearish"
    elif trend_6m > -20: return "Moderate Bearish"
    else: return "Strong Bearish"

def is_near_52_week_low(ticker: str, threshold: float = 0.7) -> bool:
    """Check if stock is near 52-week low"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current = info.get('currentPrice', info.get('regularMarketPrice', 0))
        low_52wk = info.get('fiftyTwoWeekLow', 0)
        
        return current > 0 and low_52wk > 0 and current <= low_52wk * (1 + threshold)
    except:
        return False

def get_current_price(ticker: str, is_crypto: bool = False, exchange: str = "binance") -> Dict[str, float]:
    """Get current price in USD and INR"""
    try:
        if is_crypto:
            # Try CCXT first
            try:
                ex = getattr(ccxt, exchange)()
                symbol = ticker.replace("/", "") if "/" in ticker else f"{ticker}/USDT"
                ticker_data = ex.fetch_ticker(symbol)
                price_usd = ticker_data['last']
            except:
                # Fallback to yfinance
                yf_symbol = ticker.replace("/", "-") if "/" in ticker else f"{ticker}-USD"
                stock = yf.Ticker(yf_symbol)
                hist = stock.history(period="1d")
                price_usd = hist["Close"].iloc[-1] if not hist.empty else 0
        else:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            price_usd = hist["Close"].iloc[-1] if not hist.empty else 0

        # Convert to INR
        try:
            fx = yf.Ticker("USDINR=X")
            fx_hist = fx.history(period="1d")
            inr_rate = fx_hist["Close"].iloc[-1] if not fx_hist.empty else 75.0
            price_inr = round(price_usd * inr_rate, 2)
        except:
            price_inr = round(price_usd * 75.0, 2)  # Fallback rate

        return {"usd": round(float(price_usd), 2), "inr": price_inr}

    except Exception as e:
        print(f"Error getting price for {ticker}: {e}")
        return {"usd": None, "inr": None}

def call_mistral(prompt: str) -> str:
    """Call OpenAI API for analysis"""
    if client is None:
        return "Recommendation: HOLD\nRationale: OpenAI API not configured.\nConfidence: Low"
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a financial expert analyzing stocks and crypto."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract assistant's reply correctly
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Recommendation: HOLD\nRationale: Error in LLM analysis ({str(e)}).\nConfidence: Low"

def mistral_market_analysis(ticker: str, short_df: pd.DataFrame, hist_trend: Dict, news: List, is_crypto: bool = False) -> tuple:
    """Main analysis function"""
    # Get historical data
    three_month_change = hist_trend.get("3m_change", 0)
    six_month_change = hist_trend.get("6m_change", 0)
    twelve_month_change = hist_trend.get("12m_change", 0)
    three_month_class = hist_trend.get("classification", "Neutral")
    volatility = hist_trend.get("volatility", "Low")

    # Short-term trend
    trend_arrow = "→"
    try:
        if not short_df.empty:
            # Find close column
            close_col = None
            for col in short_df.columns:
                if isinstance(col, tuple) and 'Close' in col:
                    close_col = col
                    break
                elif col == 'Close':
                    close_col = col
                    break
            
            if close_col and close_col in short_df.columns:
                closes = short_df[close_col].dropna()
                if len(closes) >= 2:
                    last_close = closes.iloc[-1]
                    prev_close = closes.iloc[-2]  # Use immediate previous close
                    pct_change = ((last_close - prev_close) / prev_close) * 100
                    trend_arrow = "↑" if pct_change > 1 else "↓" if pct_change < -1 else "→"
    except Exception as e:
        print(f"Error computing short-term trend: {e}")

    # News analysis
    news_summary = "\n".join([
        f"- {n['title']} (Relevance: {n['relevance']}, Sentiment: {n['sentiment']:.2f})"
        for n in news[:5]  # Limit to top 5
    ]) if news else "No recent headlines found."

    # Momentum strength
    momentum_strength = get_momentum_strength(three_month_change, six_month_change)
    
    # Risk context
    risk_context = ""
    if not is_crypto and is_near_52_week_low(ticker, 0.3):
        risk_context = "\n⚠️  RISK CONTEXT: Stock is trading near 52-week lows. Exercise extra caution."

    # Build prompt
    prompt = f"""
You are a financial analyst. Analyze the following {'cryptocurrency' if is_crypto else 'stock'} data for {ticker}.{risk_context}

📊 HISTORICAL PERFORMANCE:
- 3-Month Change: {three_month_change:.2f}% ({three_month_class})
- 6-Month Change: {six_month_change:.2f}% ({momentum_strength})
- 12-Month Change: {twelve_month_change:.2f}%
- Volatility: {volatility}

📉 SHORT-TERM PRICE ACTION:
Trend Arrow: {trend_arrow}

📰 NEWS HEADLINES:
{news_summary}

📋 DECISION RULES:
1. Historical trend is main driver (70-80% weight)
   - Strong bullish (>15% 6M) → BUY (downgrade to HOLD only on strong negative news)
   - Moderate bullish (5-15% 6M) → Consider BUY/HOLD
   - Weak bullish (0-5% 6M) → HOLD
   - Bearish (<0% 6M) → SELL/HOLD (upgrade only on strong positive news)

2. Short-term momentum affects confidence, not base signal

3. Only HIGH relevance news can change signals:
   - Strong negative → downgrade one step (BUY→HOLD, HOLD→SELL)
   - Strong positive → upgrade one step (SELL→HOLD, HOLD→BUY)
   - Never flip BUY↔SELL directly
   - News sentiment maybe wrongly classified, reanalyzation is must.
4. Consider absolute performance context (52-week lows, etc.)

Preliminary Classification: {'BUY' if six_month_change > 5 else 'SELL' if six_month_change < -5 else 'HOLD'}

CRYPTO-SPECIFIC RULES:
- Crypto is highly volatile → higher risk tolerance required
- 50%+ gains in 3-6 months → STRONG BUY regardless of news
- Ignore low-relevance news for crypto - focus on technicals
- High volatility is NORMAL for crypto, not a risk signal

Output format:
- Recommendation: BUY / SELL / HOLD
- Rationale: Explain with specific numbers and relevant headlines in a manner a trader would understand.
        - Include historical trend, momentum, and key news impact.
        - Do not mention rules or weights.
        - Phrase like "Based on the data, ..." or "Considering the recent trends..."
        - If a highly relevant headline influenced the decision, mention it specifically.
        
- Confidence: High / Medium / Low
"""

    # Get LLM decision
    try:
        decision = call_mistral(prompt)
    except Exception as e:
        decision = f"Recommendation: HOLD\nRationale: Error in analysis ({str(e)})\nConfidence: Low"

    return decision, trend_arrow

def parse_llm_recommendation(llm_output: str) -> Dict[str, str]:
    """More robust LLM output parsing"""
    recommendation = "HOLD"
    rationale = "Could not parse analysis"
    confidence = "Medium"
    
    # Remove markdown formatting
    clean_output = re.sub(r'\*\*', '', llm_output)
    
    # Multiple patterns for recommendation
    patterns = [
        r'Recommendation:\s*(\w+)',
        r'-\s*Recommendation:\s*(\w+)', 
        r'Signal:\s*(\w+)',
        r'Final:\s*(\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_output, re.IGNORECASE)
        if match:
            recommendation = match.group(1).upper()
            break
    
    # Extract rationale (everything between Rationale: and Confidence:)
    rationale_match = re.search(r'Rationale:(.*?)(?:Confidence:|$)', clean_output, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    
    # Extract confidence
    conf_match = re.search(r'Confidence:\s*(\w+)', clean_output, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).capitalize()
    
    return {"signal": recommendation, "rationale": rationale, "confidence": confidence}

def log_analysis_summary(result: dict, run_id: str, filename: str = "analysis_log.csv"):
    """Append a summary row to CSV (for backtesting)."""
    flat = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": result["ticker"],
        "current_price": result.get("current_price"),
        "trend": result["trend"],
        "avg_sentiment": result["avg_sentiment"],
        "ai_recommendation": result["ai_recommendation"],
        "impact_classification": result["impact_score"]["classification"],
        "impact_score": result["impact_score"]["score"],
        "final_signal": result.get("final_signal"),
        "confidence": result.get("confidence"),  # ✅ NEW: added confidence
        "hist_3m_change": result["historical_trend"].get("3m_change"),
        "hist_classification": result["historical_trend"].get("classification"),
        "hist_volatility": result["historical_trend"].get("volatility"),
        "num_news": len(result.get("news_headlines", [])),  # ✅ KEPT: with safe access
    }

    df_new = pd.DataFrame([flat])
    write_header = not os.path.isfile(filename) or os.path.getsize(filename) == 0
    df_new.to_csv(filename, mode="a", header=write_header, index=False)

    print(f"[log_analysis_summary] Logged compact row for {result['ticker']} -> {filename}")


def log_analysis_full(result: dict, run_id: str, filename: str = "analysis_log.json"):
    """Append full run to JSON log with proper array format"""
    entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": result["ticker"],
        "current_price": result.get("current_price"),
        "final_signal": result.get("final_signal"),
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "historical_trend": result["historical_trend"],
        "ai_recommendation": result["ai_recommendation"],
        "trend": result["trend"],
        "avg_sentiment": result["avg_sentiment"],
        "impact_score": result["impact_score"],
        "news_headlines": result.get("news_headlines", []),
    }

    # Read existing data or create empty list
    try:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]  # Convert single object to list
            except json.JSONDecodeError:
                data = []
    except FileNotFoundError:
        data = []

    # Append new entry
    data.append(entry)

    # Write back as proper JSON array
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"[log_analysis_full] Logged full run for {result['ticker']} -> {filename}")

# ----------------------------
# API Endpoints
# ----------------------------

class StocksRequest(BaseModel):
    tickers: List[str]
    limit: int = 5

class CryptoRequest(BaseModel):
    tickers: List[str]
    exchange: str = "binance"
    limit: int = 5

class MixedRequest(BaseModel):
    tickers: List[str]
    exchange: str = "binance"
    limit: int = 5

class AnalysisResult(BaseModel):
    run_id: str
    ticker: str
    historical_trend: Dict[str, Any]
    ai_recommendation: str
    trend: str
    avg_sentiment: float
    impact_score: Dict[str, Any]
    current_price: Dict[str, Any]
    final_signal: str
    confidence: str
    rationale: str
    news_headlines: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    dependencies: Dict[str, Any]
    system: Dict[str, Any]




import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Create a thread pool executor for blocking operations
thread_pool = ThreadPoolExecutor(max_workers=10)

async def run_in_threadpool(func, *args, **kwargs):
    """Run blocking functions in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, functools.partial(func, *args, **kwargs))

async def analyze_single_ticker_async(ticker, is_crypto=False, exchange="binance", limit=5):
    """Async wrapper for analyze_single_ticker"""
    try:
        return await run_in_threadpool(analyze_single_ticker, ticker, is_crypto, exchange, limit)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

@app.post("/analyze/stocks", tags=["Analysis"])
async def analyze_stocks(req: StocksRequest):
    """Analyze multiple stocks concurrently"""
    try:
        # Create tasks for all tickers
        tasks = [
            analyze_single_ticker_async(ticker, is_crypto=False, limit=req.limit)
            for ticker in req.tickers
        ]
        
        # Run all tasks concurrently with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300.0  # 5 minute timeout
        )
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"ticker": "unknown", "error": str(result)})
            else:
                processed_results.append(result)
                
        return processed_results
        
    except asyncio.TimeoutError:
        return [{"ticker": ticker, "error": "Analysis timeout"} for ticker in req.tickers]
    except Exception as e:
        return [{"ticker": ticker, "error": f"Server error: {str(e)}"} for ticker in req.tickers]

@app.post("/analyze/crypto", tags=["Analysis"])
async def analyze_crypto(req: CryptoRequest):
    """Analyze multiple cryptocurrencies concurrently"""
    try:
        # Create tasks for all tickers
        tasks = [
            analyze_single_ticker_async(ticker, is_crypto=True, exchange=req.exchange, limit=req.limit)
            for ticker in req.tickers
        ]
        
        # Run all tasks concurrently with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300.0  # 5 minute timeout
        )
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"ticker": "unknown", "error": str(result)})
            else:
                processed_results.append(result)
                
        return processed_results
        
    except asyncio.TimeoutError:
        return [{"ticker": ticker, "error": "Analysis timeout"} for ticker in req.tickers]
    except Exception as e:
        return [{"ticker": ticker, "error": f"Server error: {str(e)}"} for ticker in req.tickers]

@app.post("/analyze/mixed", tags=["Analysis"])
async def analyze_mixed(req: MixedRequest):
    """Analyze mixed assets concurrently"""
    try:
        tasks = []
        
        # Create tasks with appropriate parameters for each ticker
        for ticker in req.tickers:
            looks_like_crypto = "/" in ticker or ticker.endswith("USDT") or ticker.endswith("BTC")
            task = analyze_single_ticker_async(
                ticker, 
                is_crypto=looks_like_crypto, 
                exchange=req.exchange, 
                limit=req.limit
            )
            tasks.append(task)
        
        # Run all tasks concurrently with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300.0  # 5 minute timeout
        )
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"ticker": "unknown", "error": str(result)})
            else:
                processed_results.append(result)
                
        return processed_results
        
    except asyncio.TimeoutError:
        return [{"ticker": ticker, "error": "Analysis timeout"} for ticker in req.tickers]
    except Exception as e:
        return [{"ticker": ticker, "error": f"Server error: {str(e)}"} for ticker in req.tickers]
    

@app.get("/health",response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API and its dependencies"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "finbert_model_loaded": finbert_model is not None,
            "openai_available": client is not None,
            "cryptopanic_available": CRYPTOPANIC_API_KEY is not None,
            "yfinance_available": True,  # Always true as it's a library
            "ccxt_available": True,     # Always true as it's a library
        },
        "system": {
            "python_version": os.sys.version,
            "platform": os.sys.platform,
        }
    }
    
    # Test external services
    try:
        # Test Yahoo Finance
        test_stock = yf.Ticker("AAPL")
        info = test_stock.info
        status["dependencies"]["yfinance_connection"] = bool(info)
    except Exception as e:
        status["dependencies"]["yfinance_connection"] = False
        status["dependencies"]["yfinance_error"] = str(e)
    
    return status

def analyze_single_ticker(ticker: str,is_crypto:bool, exchange: str = "binance", limit: int = 5) -> Dict[str, Any]:
    """Analyze a single ticker"""
    ticker = ticker.upper().strip()
    looks_like_crypto = "/" in ticker or ticker.endswith("USDT") or ticker.endswith("BTC")
    
    if is_crypto and not looks_like_crypto:
        return {"ticker": ticker, "error": "Ticker doesn't look like a cryptocurrency"}
    
    if not is_crypto and looks_like_crypto:
        return {"ticker": ticker, "error": "Ticker looks like a cryptocurrency - use crypto endpoint"}
    
    try:
        # Get data
        if is_crypto:
            short_df = get_crypto_ohlcv(ticker, exchange, days=10)
            hist_trend = get_crypto_historical_trend(ticker, exchange=exchange)
            crypto_symbol = ticker.split("/")[0] if "/" in ticker else ticker.replace("-USD", "")
            news = get_crypto_news(crypto_symbol, limit=limit)
        else:
            short_df = get_stock_data(ticker, period="1mo", interval="1d")
            hist_trend = get_historical_trend(ticker, is_crypto=False)
            news = get_news_headlines(ticker, limit=limit)

        # Process news
        for item in news:
            item["relevance"] = classify_relevance(item["title"], is_crypto)
            item["sentiment"] = get_finbert_sentiment(item.get("content", item["title"]))

        # Calculate average sentiment
        weights = {"High": 1.0, "Medium": 0.5, "Low": 0.2}
        weighted_sentiments = sum(item["sentiment"] * weights[item["relevance"]] for item in news)
        total_weight = sum(weights[item["relevance"]] for item in news)
        avg_sentiment = weighted_sentiments / total_weight if total_weight > 0 else 0

        # Get analysis
        decision, trend = mistral_market_analysis(ticker, short_df, hist_trend, news, is_crypto)
        parsed_decision = parse_llm_recommendation(decision)
        
        # Get impact score
        impact, score = score_event(avg_sentiment, trend, hist_trend.get("classification", "Neutral"), is_crypto)
        
        # Get current price
        current_price = get_current_price(ticker, is_crypto, exchange)
        if current_price["usd"]:
            track_signal_performance(ticker, parsed_decision["signal"], current_price)
        # Build result
        run_id = str(uuid.uuid4())
        result = {
            "run_id": run_id,
            "ticker": ticker,
            "historical_trend": hist_trend,
            "ai_recommendation": decision,
            "trend": trend,
            "avg_sentiment": round(avg_sentiment, 4),
            "impact_score": {
                "classification": impact,
                "score": round(score, 4)
            },
            "current_price": current_price,
            "final_signal": parsed_decision["signal"],
            "confidence": parsed_decision["confidence"],
            "rationale": parsed_decision["rationale"],
            "news_headlines": news
        }

        # Log analysis
        log_analysis_summary(result, run_id)
        log_analysis_full(result, run_id)  

        return result

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

@app.post("/admin/run-batch-now", tags=["Performance"])
async def run_batch_now():
    """
    Manually trigger the 12-hour batch analysis job.
    Useful for debugging or forcing an immediate run.
    """
    try:
        run_batch_analysis()
        return {"status": "success", "message": "Batch analysis executed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/performance/stats",  tags=["Performance"])
def get_performance_stats(days: int = 30, min_signals: int = 5):
    """Get performance statistics for credibility dashboard"""
    try:
        # Read and parse logs
        logs = []
        try:
            with open("analysis_log.json", "r") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {"error": "No performance data available yet"}
        
        # Filter to recent period
        recent_logs = [log for log in logs if is_recent(log['timestamp'], days)]
        
        if not recent_logs:
            return {"error": f"No signals in the last {days} days"}
        
        # Calculate metrics
        total_signals = len(recent_logs)
        correct_signals = calculate_accuracy(recent_logs)
        accuracy_percentage = round((correct_signals / total_signals) * 100, 1) if total_signals > 0 else 0
        
        results = {
            "time_period": f"Last {days} days",
            "total_signals": total_signals,
            "correct_signals": correct_signals,
            "accuracy_percentage": accuracy_percentage,
            "avg_buy_gain": calculate_avg_gain(recent_logs, "BUY"),
            "avg_sell_gain": calculate_avg_gain(recent_logs, "SELL"),
            "avg_hold_performance": calculate_avg_gain(recent_logs, "HOLD"),
            "performance_by_asset": calculate_asset_performance(recent_logs),
            "signal_breakdown": {
                "BUY": sum(1 for log in recent_logs if log.get("signal") == "BUY"),
                "SELL": sum(1 for log in recent_logs if log.get("signal") == "SELL"),
                "HOLD": sum(1 for log in recent_logs if log.get("signal") == "HOLD")
            }
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Error calculating performance: {str(e)}"}



def calculate_accuracy(logs):
    """Calculate how many signals were correct with 7-day returns"""
    correct = 0
    for log in logs:
        if "future_prices" not in log or "return_7d" not in log["future_prices"]:
            continue
            
        return_7d = log["future_prices"]["return_7d"]
        signal = log.get("signal", "HOLD")
        
        # More realistic accuracy rules
        if signal == "BUY" and return_7d > 2.0:  # BUY correct if >2% gain in 7 days
            correct += 1
        elif signal == "SELL" and return_7d < -2.0:  # SELL correct if >2% drop in 7 days
            correct += 1
        elif signal == "HOLD" and abs(return_7d) <= 5.0:  # HOLD correct if within ±5%
            correct += 1
            
    return correct

# Add these imports at the top
from datetime import datetime, timedelta
import json

def is_recent(timestamp_str, days_threshold=30):
    """Check if timestamp is within last N days"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return (datetime.now() - timestamp).days <= days_threshold
    except:
        return False

def calculate_avg_gain(logs, signal_type):
    """Calculate average gain for BUY or SELL signals"""
    returns = []
    for log in logs:
        if log.get("signal") == signal_type and "return_7d" in log:
            returns.append(log["return_7d"])
    
    if not returns:
        return 0.0
    
    return round(sum(returns) / len(returns), 2)

def calculate_asset_performance(logs):
    """Calculate performance by asset type"""
    asset_stats = {}
    
    for log in logs:
        ticker = log["ticker"]
        if "return_7d" not in log:
            continue
            
        if ticker not in asset_stats:
            asset_stats[ticker] = {"total": 0, "correct": 0, "returns": []}
        
        asset_stats[ticker]["total"] += 1
        asset_stats[ticker]["returns"].append(log["return_7d"])
        
        # Check if signal was correct
        if (log["signal"] == "BUY" and log["return_7d"] > 0) or \
           (log["signal"] == "SELL" and log["return_7d"] < 0) or \
           (log["signal"] == "HOLD" and abs(log["return_7d"]) < 2):
            asset_stats[ticker]["correct"] += 1
    
    # Calculate accuracy for each asset
    result = {}
    for ticker, stats in asset_stats.items():
        accuracy = round((stats["correct"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
        avg_return = round(sum(stats["returns"]) / len(stats["returns"]), 2) if stats["returns"] else 0
        result[ticker] = {
            "accuracy": accuracy,
            "avg_return": avg_return,
            "signals": stats["total"]
        }
    
    return result

def get_price_at_date(hist_data, target_date):
    """Get closing price nearest to target date"""
    if hist_data.empty:
        return None
    
    # Find closest date in historical data
    closest_date = min(hist_data.index, key=lambda x: abs(x - target_date))
    return hist_data.loc[closest_date, 'Close']

def track_signal_performance(ticker: str, signal: str, current_price: dict):
    """Log signal performance for future analysis"""
    from datetime import datetime
    import json
    
    # Ensure we have a valid price
    price_value = current_price.get("usd") if isinstance(current_price, dict) else current_price
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "signal": signal,
        "price_at_signal": price_value,  # This field is CRITICAL
        "future_prices": {}  # Will be updated later
    }
    
    # Append to log file
    try:
        with open("analysis_log.json", "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2, default=str)
            
    except Exception as e:
        print(f"Error logging performance: {e}")
        # Create new file if doesn't exist
        with open("analysis_log.json", "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=2, default=str)

def update_future_prices():
    """Update past signals with future prices (run daily)"""
    # This would be a separate cron job
    pass

import schedule
import time
import threading
from datetime import datetime

def update_prices_job():
    """Run the price updater daily"""
    try:
        from price_updater import update_future_prices
        update_future_prices()
        print(f"Price update completed at {datetime.now()}")
    except Exception as e:
        print(f"Price update failed: {e}")

import ast

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def run_batch_analysis():
    """Run analysis for all tickers every 12 hours."""
    try:
        tickers_str = os.getenv("TICKERS", "")
        if not tickers_str:
            print("[Scheduler] No tickers found in .env")
            return
        
        tickers = [t.strip() for t in tickers_str.split(",")]
        print(f"[Scheduler] Running analysis for {len(tickers)} tickers...")

        for ticker in tickers:
            is_crypto = "/" in ticker or ticker.endswith("USDT")
            result = analyze_single_ticker(ticker, is_crypto=is_crypto)
            print(f"[Scheduler] {ticker}: {result.get('final_signal', 'N/A')}")

        print(f"[Scheduler] Batch analysis completed at {datetime.now()}")
    except Exception as e:
        print(f"[Scheduler] Batch analysis failed: {e}")

def start_scheduler():
    """Start APScheduler with 12-hour cron job."""
    scheduler = BackgroundScheduler()

    # Run every 12 hours (midnight + noon UTC)
    scheduler.add_job(run_batch_analysis, CronTrigger(hour="0,12", minute=0))

    # Keep your price updater daily at midnight
    scheduler.add_job(update_prices_job, CronTrigger(hour=0, minute=0))

    scheduler.start()
    print("[Scheduler] APScheduler started with 12-hour jobs")


