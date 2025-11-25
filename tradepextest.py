# =========================================================================================
# TRADEPEXTEST - 24/7 INTELLIGENT PAPER TRADING TESTER FOR ALL STRATEGIES
# Complete Monolithic Paper Trading System with LLM INTELLIGENCE
# Dynamically loads ALL strategies and uses AI to understand & trade them
# Version: 1.0 - FULL IMPLEMENTATION - NO PLACEHOLDERS
# =========================================================================================

"""
🚀 TRADEPEXTEST - 24/7 INTELLIGENT Paper Trading Strategy Tester

This is a COMPLETE paper trading implementation with AI INTELLIGENCE:
- Based on TradePex (2299 lines) - REAL code, NO placeholders
- **LLM INTELLIGENCE**: Uses AI to understand each strategy's unique logic
- Dynamically loads strategies from strategy library
- Pulls REAL candlestick data from HTX/Hyperliquid
- AI parses strategy rules to detect indicators needed
- AI generates trading signals by understanding entry/exit conditions
- Simulates leverage, stop loss, take profit
- Logs ALL trades and results for performance analysis
- Runs 24/7 continuously testing all strategies

THE INTELLIGENCE:
1. STRATEGY PARSER AI - Reads strategy JSON/code and extracts trading rules
2. INDICATOR DETECTOR AI - Determines which indicators each strategy needs
3. SIGNAL GENERATOR AI - Interprets strategy conditions to generate BUY/SELL signals
4. Each strategy is DIFFERENT - AI handles the differences intelligently!

Agents:
1. Strategy Loader Agent (Dynamically loads ALL strategies from library)
2. Strategy Intelligence Agent (AI understands each strategy's unique logic)
3. Market Data Agent (Fetches real candles from HTX)
4. Dynamic Indicator Engine (Calculates required indicators per strategy)
5. AI Signal Generator (Uses LLM to interpret strategy rules)
6. Paper Trading Agent (Simulates trades with leverage/SL/TP)
7. Performance Logger (Tracks results per strategy)
8. Results Analyzer (Identifies profitable strategies)

Launch: python tradepextest.py
"""

# =========================================================================================
# COMPLETE IMPORTS
# =========================================================================================

import os
import sys
import json
import time
import logging
import traceback
import threading
import queue
import signal
import hashlib
import pickle
import re
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
import asyncio

# Data processing
import numpy as np
import pandas as pd

# Environment
from dotenv import load_dotenv

# HTTP requests for market data
import requests

# LLM APIs for INTELLIGENCE
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Terminal colors
try:
    from termcolor import cprint, colored
except ImportError:
    def cprint(text, color=None, attrs=None):
        print(text)
    def colored(text, color=None, attrs=None):
        return text

# Load environment variables
load_dotenv()

# =========================================================================================
# ENHANCED LOGGING SYSTEM FOR PAPER TRADING
# =========================================================================================

def setup_papertrading_logging():
    """Setup comprehensive logging for paper trading tests"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/paper_trades", exist_ok=True)
    os.makedirs("logs/strategy_performance", exist_ok=True)
    os.makedirs("logs/ai_decisions", exist_ok=True)
    
    # Get current timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main logger
    logger = logging.getLogger("TRADEPEXTEST")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Detailed formatter
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler - Detailed logs
    file_handler = logging.FileHandler(
        f"logs/tradepextest_execution_{timestamp}.log", 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console Handler - Clean output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Specialized loggers
    components = [
        "STRATEGY-LOADER", "AI-INTELLIGENCE", "MARKET-DATA", "INDICATOR-ENGINE", 
        "SIGNAL-GENERATOR", "PAPER-TRADING", "PERFORMANCE", "RESULTS", "SYSTEM"
    ]
    
    for component in components:
        comp_logger = logging.getLogger(f"TRADEPEXTEST.{component}")
        comp_logger.setLevel(logging.INFO)
        comp_logger.addHandler(file_handler)
        comp_logger.addHandler(console_handler)
    
    return logger

# Initialize enhanced logging
logger = setup_papertrading_logging()

# =========================================================================================
# CONFIGURATION FOR PAPER TRADING
# =========================================================================================

class PaperTradingConfig:
    """Central configuration for paper trading system"""
    
    # =========================================================================================
    # PROJECT PATHS
    # =========================================================================================
    PROJECT_ROOT = Path.cwd()
    
    # Main directories
    LOGS_DIR = PROJECT_ROOT / "logs"
    PAPER_TRADES_DIR = LOGS_DIR / "paper_trades"
    STRATEGY_PERFORMANCE_DIR = LOGS_DIR / "strategy_performance"
    AI_DECISIONS_DIR = LOGS_DIR / "ai_decisions"
    
    # Strategy library paths - ALL sources
    STRATEGY_LIBRARY_DIR = PROJECT_ROOT / "strategy_library"
    APEX_CHAMPIONS_DIR = PROJECT_ROOT / "champions" / "strategies"
    SUCCESSFUL_STRATEGIES_DIR = PROJECT_ROOT / "successful_strategies"
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    MARKET_DATA_DIR = DATA_DIR / "market_data"
    
    # Results directory
    RESULTS_DIR = PROJECT_ROOT / "paper_trading_results"
    
    # =========================================================================================
    # API CONFIGURATION
    # =========================================================================================
    
    # HTX (Huobi) for market data
    HTX_BASE_URL = "https://api.huobi.pro"
    
    # Hyperliquid for additional data
    HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
    
    # LLM API Keys for INTELLIGENCE
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    
    # =========================================================================================
    # PAPER TRADING CAPITAL (SIMULATION)
    # =========================================================================================
    
    STARTING_CAPITAL_USD = 10000.0  # $10K simulated capital per strategy
    DEFAULT_LEVERAGE = 5  # 5x leverage
    MAX_POSITION_PERCENT = 0.30  # 30% max per position
    CASH_RESERVE_PERCENT = 0.20  # 20% reserve
    
    # Calculated values
    TRADEABLE_CAPITAL = STARTING_CAPITAL_USD * (1 - CASH_RESERVE_PERCENT)
    MAX_POSITION_SIZE = STARTING_CAPITAL_USD * MAX_POSITION_PERCENT
    
    # =========================================================================================
    # RISK MANAGEMENT (SIMULATION)
    # =========================================================================================
    
    DEFAULT_STOP_LOSS_PERCENT = 0.05  # 5% stop loss
    DEFAULT_TAKE_PROFIT_PERCENT = 0.15  # 15% take profit
    MAX_CONCURRENT_POSITIONS = 3
    RISK_PER_TRADE_PERCENT = 0.02  # 2% risk per trade
    
    # =========================================================================================
    # TRADING CONFIGURATION
    # =========================================================================================
    
    # Coins to trade
    TRADEABLE_COINS = [
        'BTC', 'ETH', 'SOL', 'ARB', 'MATIC', 
        'AVAX', 'OP', 'LINK', 'UNI', 'AAVE',
        'DOGE', 'XRP', 'ADA', 'DOT', 'ATOM'
    ]
    
    # Default coin if strategy doesn't specify
    DEFAULT_COIN = 'BTC'
    
    # Timeframes supported
    TIMEFRAMES = ['1m', '5m', '15m', '1H', '4H', '1D']
    DEFAULT_TIMEFRAME = '15m'
    
    # Trading fees simulation
    TRADING_FEE_PERCENT = 0.001  # 0.1% per trade
    SLIPPAGE_PERCENT = 0.001  # 0.1% slippage
    
    # =========================================================================================
    # TIMING CONFIGURATION
    # =========================================================================================
    
    STRATEGY_CHECK_INTERVAL_SECONDS = 60  # Check strategies every 1 minute
    MARKET_DATA_REFRESH_SECONDS = 30  # Refresh market data every 30 seconds
    PERFORMANCE_LOG_INTERVAL_SECONDS = 300  # Log performance every 5 minutes
    STRATEGY_RELOAD_INTERVAL_SECONDS = 600  # Reload strategies every 10 minutes
    
    @classmethod
    def ensure_all_directories(cls):
        """Create all required directories"""
        directories = [
            cls.LOGS_DIR,
            cls.PAPER_TRADES_DIR,
            cls.STRATEGY_PERFORMANCE_DIR,
            cls.AI_DECISIONS_DIR,
            cls.STRATEGY_LIBRARY_DIR,
            cls.DATA_DIR,
            cls.MARKET_DATA_DIR,
            cls.RESULTS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 All directories created/verified")

# Create all directories on load
PaperTradingConfig.ensure_all_directories()

logger.info("=" * 80)
logger.info("🚀 TRADEPEXTEST - 24/7 INTELLIGENT PAPER TRADING TESTER")
logger.info("=" * 80)
logger.info(f"Version: 1.0 (FULL IMPLEMENTATION WITH AI INTELLIGENCE)")
logger.info(f"Mode: PAPER TRADING - NO REAL MONEY")
logger.info(f"Starting Capital: ${PaperTradingConfig.STARTING_CAPITAL_USD:,.2f} per strategy")
logger.info(f"AI Intelligence: ENABLED (LLM-powered strategy understanding)")
logger.info("=" * 80)

# =========================================================================================
# GLOBAL STATE FOR PAPER TRADING
# =========================================================================================

# Strategy states - each strategy has its own paper trading account
strategy_states: Dict[str, Dict] = {}
strategy_states_lock = threading.Lock()

# Active positions per strategy
strategy_positions: Dict[str, List[Dict]] = {}
positions_lock = threading.Lock()

# Trade history per strategy
trade_history: Dict[str, List[Dict]] = {}
trade_history_lock = threading.Lock()

# Performance metrics per strategy
performance_metrics: Dict[str, Dict] = {}
performance_lock = threading.Lock()

# Market data cache
market_data_cache: Dict[str, pd.DataFrame] = {}
market_data_lock = threading.Lock()

# Parsed strategy intelligence cache
strategy_intelligence: Dict[str, Dict] = {}
intelligence_lock = threading.Lock()

# Queues for inter-thread communication
signal_queue = queue.Queue(maxsize=1000)
trade_queue = queue.Queue(maxsize=1000)
alert_queue = queue.Queue(maxsize=100)

logger.info("✅ Global state initialized")

# =========================================================================================
# DATA CLASS DEFINITIONS
# =========================================================================================

@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    strategy_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size_usd: float
    size_coins: float
    leverage: int
    stop_loss_price: float
    take_profit_price: float
    entry_time: datetime
    status: str = 'OPEN'
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    fees_paid: float = 0.0
    
@dataclass
class StrategyIntelligence:
    """AI-parsed understanding of a strategy"""
    strategy_id: str
    strategy_name: str
    strategy_type: str  # 'mean_reversion', 'momentum', 'trend_following', etc.
    required_indicators: List[str]
    entry_conditions: List[str]  # Natural language conditions
    exit_conditions: List[str]
    preferred_timeframe: str
    preferred_coins: List[str]
    stop_loss_logic: str
    take_profit_logic: str
    position_sizing: str
    ai_summary: str  # AI's understanding of the strategy
    parsed_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradingSignal:
    """Represents a trading signal from a strategy"""
    strategy_id: str
    strategy_name: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    reason: str
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5
    ai_analysis: str = ""

logger.info("✅ Data classes defined")

# =========================================================================================
# LLM MODEL FACTORY (FOR AI INTELLIGENCE)
# =========================================================================================

class ModelFactory:
    """
    Unified interface for calling different LLM providers
    Used for AI INTELLIGENCE in understanding strategies
    """
    
    @staticmethod
    def call_llm(model_config: Dict, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Universal LLM calling interface"""
        model_type = model_config.get("type", "openai")
        model_name = model_config.get("name", "gpt-4")
        
        try:
            if model_type == "deepseek":
                return ModelFactory._call_deepseek(model_name, prompt, system_prompt, temperature, max_tokens)
            elif model_type == "openai" or model_type == "gpt":
                return ModelFactory._call_openai(model_name, prompt, system_prompt, temperature, max_tokens)
            elif model_type == "anthropic" or model_type == "claude":
                return ModelFactory._call_anthropic(model_name, prompt, system_prompt, temperature, max_tokens)
            elif model_type == "xai" or model_type == "grok":
                return ModelFactory._call_xai(model_name, prompt, system_prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"❌ LLM call failed for {model_type}/{model_name}: {e}")
            raise
    
    @staticmethod
    def _call_openai(model: str, prompt: str, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Call OpenAI API"""
        if not openai:
            raise ImportError("openai package not installed")
        
        client = openai.OpenAI(api_key=PaperTradingConfig.OPENAI_API_KEY)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_anthropic(model: str, prompt: str, system_prompt: Optional[str],
                       temperature: float, max_tokens: int) -> str:
        """Call Anthropic API"""
        if not anthropic:
            raise ImportError("anthropic package not installed")
        
        client = anthropic.Anthropic(api_key=PaperTradingConfig.ANTHROPIC_API_KEY)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "",
            messages=messages
        )
        
        return response.content[0].text
    
    @staticmethod
    def _call_deepseek(model: str, prompt: str, system_prompt: Optional[str],
                      temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API (OpenAI-compatible)"""
        if not openai:
            raise ImportError("openai package not installed")
        
        client = openai.OpenAI(
            api_key=PaperTradingConfig.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_xai(model: str, prompt: str, system_prompt: Optional[str],
                 temperature: float, max_tokens: int) -> str:
        """Call xAI Grok API (OpenAI-compatible)"""
        if not openai:
            raise ImportError("openai package not installed")
        
        client = openai.OpenAI(
            api_key=PaperTradingConfig.XAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def get_available_model() -> Dict:
        """Get first available model based on configured API keys"""
        if PaperTradingConfig.XAI_API_KEY:
            return {"type": "xai", "name": "grok-3-mini-fast"}
        elif PaperTradingConfig.DEEPSEEK_API_KEY:
            return {"type": "deepseek", "name": "deepseek-chat"}
        elif PaperTradingConfig.OPENAI_API_KEY:
            return {"type": "openai", "name": "gpt-4o-mini"}
        elif PaperTradingConfig.ANTHROPIC_API_KEY:
            return {"type": "anthropic", "name": "claude-3-haiku-20240307"}
        else:
            return None

logger.info("✅ Model Factory initialized (LLM Intelligence Layer)")

# =========================================================================================
# HTX MARKET DATA FETCHER (REAL DATA!)
# =========================================================================================

class HTXMarketDataFetcher:
    """
    Fetches REAL market data from HTX (Huobi) exchange
    Provides candlestick data for paper trading simulations
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.MARKET-DATA")
        self.base_url = PaperTradingConfig.HTX_BASE_URL
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 30  # 30 seconds cache
        
    def fetch_candles(self, symbol: str = 'btcusdt', period: str = '15min', 
                      count: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candlestick data from HTX
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt', 'ethusdt')
            period: Candle period (1min, 5min, 15min, 60min, 4hour, 1day)
            count: Number of candles (max 2000)
            
        Returns:
            DataFrame with columns: datetime, Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            if time.time() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key].copy()
        
        try:
            self.logger.debug(f"📊 Fetching {symbol.upper()} {period} candles from HTX...")
            
            endpoint = f"{self.base_url}/market/history/kline"
            params = {
                "symbol": symbol.lower(),
                "period": period,
                "size": min(count, 2000)
            }
            
            response = requests.get(endpoint, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"HTX API error: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get("status") != "ok":
                self.logger.error(f"HTX API error: {data.get('err-msg', 'Unknown')}")
                return None
            
            klines = data.get("data", [])
            
            if not klines:
                self.logger.warning(f"No data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines)
            
            # Rename columns to standard OHLCV
            df = df.rename(columns={
                'id': 'timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'amount': 'Volume'
            })
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Select and reorder columns
            df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Sort by datetime (oldest first)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Cache the result
            self.cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.cache_ttl
            
            self.logger.debug(f"✅ Fetched {len(df)} candles for {symbol.upper()}")
            
            return df.copy()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            return None
    
    def get_current_price(self, symbol: str = 'btcusdt') -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            endpoint = f"{self.base_url}/market/trade"
            params = {"symbol": symbol.lower()}
            
            response = requests.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    trades = data.get("tick", {}).get("data", [])
                    if trades:
                        return float(trades[0]["price"])
            
            # Fallback to candle close price
            candles = self.fetch_candles(symbol, '1min', 1)
            if candles is not None and len(candles) > 0:
                return float(candles['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_multiple_coins_data(self, coins: List[str], period: str = '15min', 
                                 count: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch candle data for multiple coins"""
        result = {}
        
        for coin in coins:
            symbol = f"{coin.lower()}usdt"
            df = self.fetch_candles(symbol, period, count)
            if df is not None:
                result[coin] = df
            time.sleep(0.1)  # Rate limiting
        
        return result

# Initialize global market data fetcher
market_data_fetcher = HTXMarketDataFetcher()
logger.info("✅ HTX Market Data Fetcher initialized")

# =========================================================================================
# STRATEGY INTELLIGENCE AGENT (AI UNDERSTANDS EACH STRATEGY!)
# =========================================================================================

class StrategyIntelligenceAgent:
    """
    AI-powered agent that UNDERSTANDS each strategy
    Uses LLM to parse strategy rules, detect indicators, and generate signals
    
    THIS IS THE BRAIN - It doesn't know the strategy beforehand,
    it analyzes each one dynamically!
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.AI-INTELLIGENCE")
        self.parsed_strategies: Dict[str, StrategyIntelligence] = {}
        self.model = ModelFactory.get_available_model()
        
        if self.model:
            self.logger.info(f"🧠 AI Intelligence using: {self.model['type']}/{self.model['name']}")
        else:
            self.logger.warning("⚠️ No LLM API key found - using rule-based fallback")
    
    def parse_strategy(self, strategy_data: Dict) -> StrategyIntelligence:
        """
        Use AI to understand a strategy that we've never seen before
        
        Args:
            strategy_data: Raw strategy dictionary from JSON file
            
        Returns:
            StrategyIntelligence object with AI's understanding
        """
        strategy_id = strategy_data.get('name', 'unknown').replace(' ', '_').lower()
        
        # Check cache
        if strategy_id in self.parsed_strategies:
            return self.parsed_strategies[strategy_id]
        
        self.logger.info(f"🧠 AI analyzing strategy: {strategy_data.get('name', 'Unknown')}")
        
        if self.model:
            intelligence = self._ai_parse_strategy(strategy_data)
        else:
            intelligence = self._fallback_parse_strategy(strategy_data)
        
        # Cache it
        self.parsed_strategies[strategy_id] = intelligence
        
        self.logger.info(f"✅ Strategy understood: {intelligence.strategy_type}")
        self.logger.info(f"   Indicators needed: {intelligence.required_indicators}")
        self.logger.info(f"   Timeframe: {intelligence.preferred_timeframe}")
        
        return intelligence
    
    def _ai_parse_strategy(self, strategy_data: Dict) -> StrategyIntelligence:
        """Use LLM to parse and understand strategy"""
        
        # Prepare strategy text for AI analysis
        strategy_text = json.dumps(strategy_data, indent=2, default=str)
        
        system_prompt = """You are an expert trading strategy analyst. 
Your job is to analyze trading strategies and extract key information.
Be precise and extract exactly what the strategy needs to function.
Respond ONLY with valid JSON, no other text."""
        
        user_prompt = f"""Analyze this trading strategy and extract its requirements:

STRATEGY DATA:
{strategy_text}

Extract and return a JSON object with these fields:
{{
    "strategy_type": "one of: mean_reversion, momentum, trend_following, breakout, scalping, market_making, pairs_trading, volatility",
    "required_indicators": ["list of indicators needed like: rsi, macd, ema, sma, bollinger, vwap, atr, stochastic, cci, adx, obv, mfi"],
    "entry_conditions": ["list of entry conditions in plain english"],
    "exit_conditions": ["list of exit conditions in plain english"],
    "preferred_timeframe": "15m or 1H or 4H or 1D",
    "preferred_coins": ["BTC", "ETH", "SOL"],
    "stop_loss_percent": 5,
    "take_profit_percent": 15,
    "position_sizing": "description of position sizing",
    "ai_summary": "2-3 sentence summary of what this strategy does"
}}

Return ONLY the JSON object."""

        try:
            response = ModelFactory.call_llm(
                self.model,
                user_prompt,
                system_prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            # Parse JSON from response
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)
            
            return StrategyIntelligence(
                strategy_id=strategy_data.get('name', 'unknown').replace(' ', '_').lower(),
                strategy_name=strategy_data.get('name', 'Unknown Strategy'),
                strategy_type=parsed.get('strategy_type', 'unknown'),
                required_indicators=parsed.get('required_indicators', ['ema', 'rsi']),
                entry_conditions=parsed.get('entry_conditions', []),
                exit_conditions=parsed.get('exit_conditions', []),
                preferred_timeframe=parsed.get('preferred_timeframe', '15m'),
                preferred_coins=parsed.get('preferred_coins', ['BTC']),
                stop_loss_logic=f"{parsed.get('stop_loss_percent', 5)}% stop loss",
                take_profit_logic=f"{parsed.get('take_profit_percent', 15)}% take profit",
                position_sizing=parsed.get('position_sizing', '30% of capital'),
                ai_summary=parsed.get('ai_summary', 'Strategy analyzed by AI')
            )
            
        except Exception as e:
            self.logger.error(f"AI parsing failed: {e}")
            return self._fallback_parse_strategy(strategy_data)
    
    def _fallback_parse_strategy(self, strategy_data: Dict) -> StrategyIntelligence:
        """Fallback rule-based parsing when LLM is not available"""
        
        # Convert all strategy text to lowercase for matching
        text = json.dumps(strategy_data, default=str).lower()
        
        # Detect strategy type
        strategy_type = 'unknown'
        if 'mean reversion' in text or 'vwap' in text:
            strategy_type = 'mean_reversion'
        elif 'momentum' in text or 'rsi' in text:
            strategy_type = 'momentum'
        elif 'trend' in text or 'ema' in text or 'sma' in text:
            strategy_type = 'trend_following'
        elif 'breakout' in text or 'bollinger' in text:
            strategy_type = 'breakout'
        elif 'market making' in text or 'stoikov' in text or 'inventory' in text:
            strategy_type = 'market_making'
        elif 'pairs' in text or 'cointegration' in text:
            strategy_type = 'pairs_trading'
        
        # Detect required indicators
        indicators = []
        indicator_keywords = {
            'rsi': ['rsi', 'relative strength'],
            'macd': ['macd'],
            'ema': ['ema', 'exponential moving'],
            'sma': ['sma', 'simple moving'],
            'bollinger': ['bollinger', 'bb_'],
            'vwap': ['vwap', 'volume weighted'],
            'atr': ['atr', 'average true range'],
            'stochastic': ['stochastic', 'stoch'],
            'cci': ['cci', 'commodity channel'],
            'adx': ['adx', 'directional'],
            'obv': ['obv', 'on balance'],
            'mfi': ['mfi', 'money flow'],
            'pivot': ['pivot', 'support', 'resistance'],
            'ichimoku': ['ichimoku', 'tenkan', 'kijun'],
        }
        
        for indicator, keywords in indicator_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    indicators.append(indicator)
                    break
        
        if not indicators:
            indicators = ['ema', 'rsi']  # Default indicators
        
        return StrategyIntelligence(
            strategy_id=strategy_data.get('name', 'unknown').replace(' ', '_').lower(),
            strategy_name=strategy_data.get('name', 'Unknown Strategy'),
            strategy_type=strategy_type,
            required_indicators=list(set(indicators)),
            entry_conditions=[strategy_data.get('entry_rules', 'Not specified')],
            exit_conditions=[strategy_data.get('exit_rules', 'Not specified')],
            preferred_timeframe=strategy_data.get('timeframe', '15m'),
            preferred_coins=strategy_data.get('assets', ['BTC']),
            stop_loss_logic=strategy_data.get('stop_loss', '5% stop loss'),
            take_profit_logic=strategy_data.get('take_profit', '15% take profit'),
            position_sizing=strategy_data.get('position_sizing', '30% of capital'),
            ai_summary=f"Rule-based parsing of {strategy_type} strategy"
        )

logger.info("✅ Strategy Intelligence Agent defined (AI-powered strategy understanding)")

# =========================================================================================
# AI SWARM DECISION MAKER (MULTIPLE LLMs VOTE!)
# =========================================================================================

class SwarmDecisionMaker:
    """
    AI SWARM - Multiple LLMs vote on trading decisions!
    Just like original tradepex.py but for paper trading
    
    Consensus mechanism:
    - Query 2-3 different LLMs
    - Each votes: APPROVE or REJECT
    - Majority wins
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.AI-SWARM")
        self.available_models = self._detect_available_models()
        self.logger.info(f"�� AI Swarm initialized with {len(self.available_models)} models")
        
    def _detect_available_models(self) -> List[Dict]:
        """Detect which LLM models are available based on API keys"""
        models = []
        
        if PaperTradingConfig.XAI_API_KEY:
            models.append({"type": "xai", "name": "grok-3-mini-fast", "label": "Grok"})
        
        if PaperTradingConfig.DEEPSEEK_API_KEY:
            models.append({"type": "deepseek", "name": "deepseek-chat", "label": "DeepSeek"})
        
        if PaperTradingConfig.OPENAI_API_KEY:
            models.append({"type": "openai", "name": "gpt-4o-mini", "label": "GPT-4"})
        
        if PaperTradingConfig.ANTHROPIC_API_KEY:
            models.append({"type": "anthropic", "name": "claude-3-haiku-20240307", "label": "Claude"})
        
        return models[:3]  # Max 3 models for swarm
    
    def get_swarm_consensus(self, strategy_name: str, symbol: str, 
                            direction: str, size_usd: float, 
                            reason: str, indicators: Dict,
                            current_price: float) -> Dict:
        """
        Get AI Swarm consensus for a trading decision
        
        Returns:
            {
                'decision': 'APPROVE' or 'REJECT',
                'votes': {'Grok': 'APPROVE', 'DeepSeek': 'REJECT', ...},
                'reason': 'Consensus reason',
                'recommended_size': float,
                'confidence': float
            }
        """
        
        if len(self.available_models) == 0:
            self.logger.warning("⚠️ No LLM models available - auto-approving for paper trading")
            return {
                'decision': 'APPROVE',
                'votes': {'fallback': 'APPROVE'},
                'reason': 'No LLM available - auto-approved for paper trading',
                'recommended_size': size_usd,
                'confidence': 0.5
            }
        
        self.logger.info(f"🤖 Requesting AI Swarm consensus for {symbol} {direction}...")
        
        # Prepare the prompt
        prompt = self._create_consensus_prompt(
            strategy_name, symbol, direction, size_usd, 
            reason, indicators, current_price
        )
        
        # Query each model
        votes = {}
        recommendations = []
        
        for model in self.available_models:
            try:
                response = self._query_model(model, prompt)
                vote, rec_size, model_reason = self._parse_vote(response, size_usd)
                votes[model['label']] = vote
                recommendations.append(rec_size)
                
                self.logger.info(f"   {model['label']}: {vote} (size: ${rec_size:.2f})")
                
            except Exception as e:
                self.logger.warning(f"   {model['label']}: FAILED - {e}")
                votes[model['label']] = 'ABSTAIN'
        
        # Calculate consensus
        approve_count = sum(1 for v in votes.values() if v == 'APPROVE')
        reject_count = sum(1 for v in votes.values() if v == 'REJECT')
        total_votes = approve_count + reject_count
        
        if total_votes == 0:
            decision = 'APPROVE'  # Default approve for paper trading
            confidence = 0.5
        else:
            decision = 'APPROVE' if approve_count > reject_count else 'REJECT'
            confidence = max(approve_count, reject_count) / total_votes
        
        # Average recommended size
        recommended_size = np.mean(recommendations) if recommendations else size_usd
        
        result = {
            'decision': decision,
            'votes': votes,
            'reason': f"Swarm voted {approve_count}-{reject_count}",
            'recommended_size': recommended_size,
            'confidence': confidence
        }
        
        self.logger.info(f"🎯 Swarm Decision: {decision} (confidence: {confidence:.0%})")
        
        return result
    
    def _create_consensus_prompt(self, strategy_name: str, symbol: str,
                                  direction: str, size_usd: float,
                                  reason: str, indicators: Dict,
                                  current_price: float) -> str:
        """Create prompt for swarm voting"""
        
        # Format indicators for display
        indicator_text = "\n".join([f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}" 
                                    for k, v in indicators.items()])
        
        return f"""You are an AI trading advisor. Evaluate this PAPER TRADING signal:

TRADE PROPOSAL:
- Strategy: {strategy_name}
- Symbol: {symbol}
- Current Price: ${current_price:.2f}
- Direction: {direction}
- Proposed Size: ${size_usd:.2f}
- Reason: {reason}

CURRENT INDICATORS:
{indicator_text}

CONTEXT:
- This is PAPER TRADING (simulated, no real money)
- We want to test if this strategy would work in real conditions
- The goal is to log results and see which strategies are profitable

EVALUATE:
1. Does the trading signal make sense given the indicators?
2. Is the direction (BUY/SELL) appropriate for current conditions?
3. What position size would you recommend?

Respond with EXACTLY this format:
DECISION: APPROVE or REJECT
RECOMMENDED_SIZE: $XXX
REASON: Your brief reason (max 30 words)"""
    
    def _query_model(self, model: Dict, prompt: str) -> str:
        """Query a single model"""
        system_prompt = "You are an AI trading advisor. Be concise and direct."
        
        return ModelFactory.call_llm(
            model,
            prompt,
            system_prompt,
            temperature=0.2,
            max_tokens=200
        )
    
    def _parse_vote(self, response: str, default_size: float) -> Tuple[str, float, str]:
        """Parse model response into vote, size, reason"""
        
        response_upper = response.upper()
        
        # Parse decision
        if 'APPROVE' in response_upper:
            vote = 'APPROVE'
        elif 'REJECT' in response_upper:
            vote = 'REJECT'
        else:
            vote = 'ABSTAIN'
        
        # Parse recommended size
        size_match = re.search(r'RECOMMENDED_SIZE:\s*\$?([\d,.]+)', response, re.IGNORECASE)
        if size_match:
            try:
                rec_size = float(size_match.group(1).replace(',', ''))
            except:
                rec_size = default_size
        else:
            rec_size = default_size
        
        # Parse reason
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        
        return vote, rec_size, reason

logger.info("✅ AI Swarm Decision Maker defined (Multi-LLM voting system)")

# =========================================================================================
# DYNAMIC INDICATOR ENGINE (CALCULATES ALL INDICATORS)
# =========================================================================================

class DynamicIndicatorEngine:
    """
    Dynamically calculates indicators based on what each strategy needs
    Each strategy is different - this engine handles all of them!
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.INDICATOR-ENGINE")
        
    def calculate_all_indicators(self, df: pd.DataFrame, 
                                  required_indicators: List[str]) -> Dict[str, Any]:
        """
        Calculate all required indicators for the given market data
        
        Args:
            df: DataFrame with OHLCV data
            required_indicators: List of indicator names needed by strategy
            
        Returns:
            Dictionary with all indicator values
        """
        indicators = {}
        
        if df is None or len(df) < 20:
            self.logger.warning("Insufficient data for indicator calculation")
            return indicators
        
        try:
            # Always include current price info
            indicators['current_price'] = float(df['Close'].iloc[-1])
            indicators['prev_close'] = float(df['Close'].iloc[-2]) if len(df) > 1 else indicators['current_price']
            indicators['high'] = float(df['High'].iloc[-1])
            indicators['low'] = float(df['Low'].iloc[-1])
            indicators['volume'] = float(df['Volume'].iloc[-1])
            indicators['price_change_pct'] = (indicators['current_price'] - indicators['prev_close']) / indicators['prev_close'] * 100
            
            for indicator in required_indicators:
                try:
                    indicator = indicator.lower()
                    if indicator == 'rsi':
                        indicators.update(self._calculate_rsi(df))
                    elif indicator == 'macd':
                        indicators.update(self._calculate_macd(df))
                    elif indicator == 'sma':
                        indicators.update(self._calculate_sma(df))
                    elif indicator == 'ema':
                        indicators.update(self._calculate_ema(df))
                    elif indicator == 'bollinger':
                        indicators.update(self._calculate_bollinger(df))
                    elif indicator == 'atr':
                        indicators.update(self._calculate_atr(df))
                    elif indicator == 'vwap':
                        indicators.update(self._calculate_vwap(df))
                    elif indicator == 'stochastic':
                        indicators.update(self._calculate_stochastic(df))
                    elif indicator == 'cci':
                        indicators.update(self._calculate_cci(df))
                    elif indicator == 'adx':
                        indicators.update(self._calculate_adx(df))
                    elif indicator == 'obv':
                        indicators.update(self._calculate_obv(df))
                    elif indicator == 'mfi':
                        indicators.update(self._calculate_mfi(df))
                    elif indicator == 'momentum':
                        indicators.update(self._calculate_momentum(df))
                    elif indicator == 'pivot':
                        indicators.update(self._calculate_pivot_points(df))
                    elif indicator == 'ichimoku':
                        indicators.update(self._calculate_ichimoku(df))
                except Exception as e:
                    self.logger.debug(f"Error calculating {indicator}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in indicator calculation: {e}")
        
        return indicators
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate RSI"""
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            'rsi_prev': float(rsi.iloc[-2]) if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else 50.0
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD"""
        close = df['Close']
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'macd_signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            'macd_histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        }
    
    def _calculate_sma(self, df: pd.DataFrame) -> Dict:
        """Calculate SMAs"""
        close = df['Close']
        result = {}
        for period in [10, 20, 50, 100, 200]:
            if len(close) >= period:
                sma = close.rolling(window=period).mean()
                result[f'sma_{period}'] = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else float(close.iloc[-1])
        return result
    
    def _calculate_ema(self, df: pd.DataFrame) -> Dict:
        """Calculate EMAs"""
        close = df['Close']
        result = {}
        for period in [9, 12, 20, 26, 50, 100, 200]:
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean()
                result[f'ema_{period}'] = float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else float(close.iloc[-1])
        return result
    
    def _calculate_bollinger(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calculate Bollinger Bands"""
        close = df['Close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        current_price = float(close.iloc[-1])
        return {
            'bb_upper': float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else current_price * 1.02,
            'bb_middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else current_price,
            'bb_lower': float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else current_price * 0.98,
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate ATR"""
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return {
            'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            'atr_percent': float(atr.iloc[-1] / close.iloc[-1] * 100) if not pd.isna(atr.iloc[-1]) else 0.0
        }
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Dict:
        """Calculate VWAP"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        current_price = float(df['Close'].iloc[-1])
        vwap_value = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else current_price
        std = (typical_price - vwap).std()
        return {
            'vwap': vwap_value,
            'vwap_upper': vwap_value + (2 * std) if not pd.isna(std) else vwap_value * 1.02,
            'vwap_lower': vwap_value - (2 * std) if not pd.isna(std) else vwap_value * 0.98,
            'vwap_deviation': (current_price - vwap_value) / vwap_value * 100 if vwap_value != 0 else 0
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic"""
        high, low, close = df['High'], df['Low'], df['Close']
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return {
            'stoch_k': float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0,
            'stoch_d': float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0
        }
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """Calculate CCI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = (typical_price - sma).abs().rolling(window=period).mean()
        cci = (typical_price - sma) / (0.015 * mad)
        return {'cci': float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0}
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate ADX"""
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return {
            'adx': float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 25.0,
            'plus_di': float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 25.0,
            'minus_di': float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 25.0
        }
    
    def _calculate_obv(self, df: pd.DataFrame) -> Dict:
        """Calculate OBV"""
        close, volume = df['Close'], df['Volume']
        obv = (np.sign(close.diff()) * volume).cumsum()
        return {
            'obv': float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0.0,
            'obv_sma': float(obv.rolling(window=20).mean().iloc[-1]) if len(obv) >= 20 else float(obv.iloc[-1])
        }
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate MFI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return {'mfi': float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0}
    
    def _calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> Dict:
        """Calculate Momentum"""
        close = df['Close']
        momentum = close - close.shift(period)
        return {
            'momentum': float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0.0,
            'momentum_percent': float(momentum.iloc[-1] / close.shift(period).iloc[-1] * 100) if not pd.isna(momentum.iloc[-1]) else 0.0
        }
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate Pivot Points"""
        high = float(df['High'].iloc[-2]) if len(df) > 1 else float(df['High'].iloc[-1])
        low = float(df['Low'].iloc[-2]) if len(df) > 1 else float(df['Low'].iloc[-1])
        close = float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Close'].iloc[-1])
        pivot = (high + low + close) / 3
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low, 'r2': pivot + (high - low),
            's1': 2 * pivot - high, 's2': pivot - (high - low)
        }
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku"""
        high, low, close = df['High'], df['Low'], df['Close']
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2
        return {
            'ichimoku_tenkan': float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else float(close.iloc[-1]),
            'ichimoku_kijun': float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else float(close.iloc[-1]),
            'ichimoku_senkou_a': float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else float(close.iloc[-1]),
            'ichimoku_senkou_b': float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else float(close.iloc[-1])
        }

# Initialize indicator engine
indicator_engine = DynamicIndicatorEngine()
logger.info("✅ Dynamic Indicator Engine initialized")

# =========================================================================================
# AI SIGNAL GENERATOR (GENERATES BUY/SELL SIGNALS WITH INTELLIGENCE)
# =========================================================================================

class AISignalGenerator:
    """
    Uses AI to generate trading signals by understanding strategy rules
    Each strategy is different - AI interprets the conditions!
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.SIGNAL-GENERATOR")
        self.model = ModelFactory.get_available_model()
        
    def generate_signal(self, strategy_intelligence: StrategyIntelligence,
                        indicators: Dict, has_position: bool,
                        current_position_direction: Optional[str] = None) -> TradingSignal:
        """
        Generate a trading signal using AI to interpret strategy rules
        
        Args:
            strategy_intelligence: AI's understanding of the strategy
            indicators: Calculated indicator values
            has_position: Whether we currently have a position
            current_position_direction: 'LONG' or 'SHORT' if has_position
            
        Returns:
            TradingSignal with action and reason
        """
        
        if self.model:
            return self._ai_generate_signal(strategy_intelligence, indicators, 
                                             has_position, current_position_direction)
        else:
            return self._rule_based_signal(strategy_intelligence, indicators,
                                            has_position, current_position_direction)
    
    def _ai_generate_signal(self, intelligence: StrategyIntelligence,
                            indicators: Dict, has_position: bool,
                            current_direction: Optional[str]) -> TradingSignal:
        """Use AI to generate trading signal"""
        
        # Format indicators for AI
        indicator_text = "\n".join([f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                                    for k, v in indicators.items()])
        
        entry_conditions = "\n".join([f"  - {c}" for c in intelligence.entry_conditions])
        exit_conditions = "\n".join([f"  - {c}" for c in intelligence.exit_conditions])
        
        system_prompt = """You are an AI trading signal generator.
Analyze the strategy rules and current market indicators to decide: BUY, SELL, or HOLD.
Be precise and follow the strategy's rules exactly.
Respond ONLY with valid JSON."""
        
        user_prompt = f"""Analyze this trading situation and generate a signal:

STRATEGY: {intelligence.strategy_name}
TYPE: {intelligence.strategy_type}

ENTRY CONDITIONS:
{entry_conditions}

EXIT CONDITIONS:
{exit_conditions}

CURRENT INDICATORS:
{indicator_text}

CURRENT POSITION: {"Has " + current_direction + " position" if has_position else "No position"}

Based on the strategy rules and current indicators, what should we do?

Respond with JSON:
{{
    "action": "BUY" or "SELL" or "HOLD",
    "reason": "Brief explanation of why (max 50 words)",
    "confidence": 0.0 to 1.0
}}"""

        try:
            response = ModelFactory.call_llm(
                self.model,
                user_prompt,
                system_prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)
            
            return TradingSignal(
                strategy_id=intelligence.strategy_id,
                strategy_name=intelligence.strategy_name,
                symbol=intelligence.preferred_coins[0] if intelligence.preferred_coins else 'BTC',
                action=parsed.get('action', 'HOLD'),
                reason=parsed.get('reason', 'AI decision'),
                indicators=indicators,
                confidence=parsed.get('confidence', 0.5),
                ai_analysis=response
            )
            
        except Exception as e:
            self.logger.error(f"AI signal generation failed: {e}")
            return self._rule_based_signal(intelligence, indicators, has_position, current_direction)
    
    def _rule_based_signal(self, intelligence: StrategyIntelligence,
                           indicators: Dict, has_position: bool,
                           current_direction: Optional[str]) -> TradingSignal:
        """Fallback rule-based signal generation"""
        
        action = 'HOLD'
        reason = 'No clear signal'
        confidence = 0.5
        
        current_price = indicators.get('current_price', 0)
        
        # Strategy-type specific logic
        if intelligence.strategy_type == 'mean_reversion':
            # Mean reversion logic
            if 'vwap' in indicators:
                vwap = indicators['vwap']
                vwap_lower = indicators.get('vwap_lower', vwap * 0.98)
                vwap_upper = indicators.get('vwap_upper', vwap * 1.02)
                
                if not has_position:
                    if current_price < vwap_lower:
                        action, reason = 'BUY', f'Price ${current_price:.2f} below VWAP lower band (mean reversion buy)'
                        confidence = 0.7
                    elif current_price > vwap_upper:
                        action, reason = 'SELL', f'Price ${current_price:.2f} above VWAP upper band (mean reversion sell)'
                        confidence = 0.7
                else:
                    if abs(current_price - vwap) / vwap < 0.005:
                        action, reason = 'SELL', f'Price returned to VWAP (mean reversion complete)'
                        confidence = 0.6
            
            elif 'rsi' in indicators:
                rsi = indicators['rsi']
                if not has_position:
                    if rsi < 30:
                        action, reason = 'BUY', f'RSI {rsi:.1f} oversold (mean reversion buy)'
                        confidence = 0.7
                    elif rsi > 70:
                        action, reason = 'SELL', f'RSI {rsi:.1f} overbought (mean reversion sell)'
                        confidence = 0.7
        
        elif intelligence.strategy_type == 'momentum':
            # Momentum logic
            rsi = indicators.get('rsi', 50)
            macd_hist = indicators.get('macd_histogram', 0)
            
            if not has_position:
                if rsi > 50 and macd_hist > 0:
                    action, reason = 'BUY', f'Bullish momentum: RSI {rsi:.1f}, MACD histogram positive'
                    confidence = 0.65
                elif rsi < 50 and macd_hist < 0:
                    action, reason = 'SELL', f'Bearish momentum: RSI {rsi:.1f}, MACD histogram negative'
                    confidence = 0.65
        
        elif intelligence.strategy_type == 'trend_following':
            # Trend following logic
            ema_20 = indicators.get('ema_20', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            
            if not has_position:
                if ema_20 > ema_50:
                    action, reason = 'BUY', f'Bullish trend: EMA20 > EMA50'
                    confidence = 0.6
                elif ema_20 < ema_50:
                    action, reason = 'SELL', f'Bearish trend: EMA20 < EMA50'
                    confidence = 0.6
            else:
                # Exit on trend reversal
                if current_direction == 'LONG' and ema_20 < ema_50:
                    action, reason = 'SELL', 'Trend reversal: exit long'
                    confidence = 0.65
                elif current_direction == 'SHORT' and ema_20 > ema_50:
                    action, reason = 'BUY', 'Trend reversal: cover short'
                    confidence = 0.65
        
        elif intelligence.strategy_type == 'breakout':
            # Breakout logic
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            bb_width = indicators.get('bb_width', 0.04)
            
            if not has_position:
                if current_price > bb_upper and bb_width < 0.03:
                    action, reason = 'BUY', f'Breakout above Bollinger band after squeeze'
                    confidence = 0.7
                elif current_price < bb_lower and bb_width < 0.03:
                    action, reason = 'SELL', f'Breakdown below Bollinger band after squeeze'
                    confidence = 0.7
        
        elif intelligence.strategy_type == 'market_making':
            # Market making / inventory management logic
            if not has_position:
                # Look for good entry on bid side
                if 'vwap_deviation' in indicators:
                    deviation = indicators['vwap_deviation']
                    if deviation < -1:
                        action, reason = 'BUY', f'Below VWAP by {abs(deviation):.2f}% - market making buy'
                        confidence = 0.6
                    elif deviation > 1:
                        action, reason = 'SELL', f'Above VWAP by {deviation:.2f}% - market making sell'
                        confidence = 0.6
        
        else:
            # Generic fallback
            rsi = indicators.get('rsi', 50)
            if not has_position:
                if rsi < 30:
                    action, reason = 'BUY', f'Oversold RSI {rsi:.1f}'
                    confidence = 0.55
                elif rsi > 70:
                    action, reason = 'SELL', f'Overbought RSI {rsi:.1f}'
                    confidence = 0.55
        
        return TradingSignal(
            strategy_id=intelligence.strategy_id,
            strategy_name=intelligence.strategy_name,
            symbol=intelligence.preferred_coins[0] if intelligence.preferred_coins else 'BTC',
            action=action,
            reason=reason,
            indicators=indicators,
            confidence=confidence,
            ai_analysis="Rule-based signal"
        )

# Initialize signal generator
signal_generator = AISignalGenerator()
logger.info("✅ AI Signal Generator initialized")

# =========================================================================================
# STRATEGY LOADER AGENT (LOADS ALL STRATEGIES DYNAMICALLY)
# =========================================================================================

class StrategyLoaderAgent:
    """
    Dynamically loads ALL strategies from the strategy library
    Each strategy is different - handles JSON and Python files
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.STRATEGY-LOADER")
        self.loaded_strategies: Dict[str, Dict] = {}
        self.strategy_intelligence: Dict[str, StrategyIntelligence] = {}
        self.intelligence_agent = StrategyIntelligenceAgent()
        
    def load_all_strategies(self) -> Dict[str, Dict]:
        """Load all strategies from all sources"""
        self.logger.info("📂 Loading all strategies...")
        
        strategies = {}
        
        # Load from strategy library
        if PaperTradingConfig.STRATEGY_LIBRARY_DIR.exists():
            lib_strategies = self._load_from_directory(
                PaperTradingConfig.STRATEGY_LIBRARY_DIR, 
                "strategy_library"
            )
            strategies.update(lib_strategies)
        
        # Load from champions
        if PaperTradingConfig.APEX_CHAMPIONS_DIR.exists():
            champ_strategies = self._load_from_directory(
                PaperTradingConfig.APEX_CHAMPIONS_DIR,
                "champions"
            )
            strategies.update(champ_strategies)
        
        # Load from successful strategies
        if PaperTradingConfig.SUCCESSFUL_STRATEGIES_DIR.exists():
            success_strategies = self._load_from_directory(
                PaperTradingConfig.SUCCESSFUL_STRATEGIES_DIR,
                "successful"
            )
            strategies.update(success_strategies)
        
        self.loaded_strategies = strategies
        self.logger.info(f"✅ Loaded {len(strategies)} strategies total")
        
        # Parse each strategy with AI
        self._parse_all_strategies()
        
        return strategies
    
    def _load_from_directory(self, directory: Path, source: str) -> Dict[str, Dict]:
        """Load strategies from a directory"""
        strategies = {}
        
        try:
            # Load JSON files
            for json_file in directory.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        strategy_data = json.load(f)
                    
                    strategy_id = json_file.stem
                    strategy_data['_source'] = source
                    strategy_data['_file'] = str(json_file)
                    strategies[strategy_id] = strategy_data
                    
                    self.logger.debug(f"   Loaded: {strategy_id}")
                    
                except Exception as e:
                    self.logger.warning(f"   Failed to load {json_file.name}: {e}")
            
            # Also check for Python strategy files
            for py_file in directory.glob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        code = f.read()
                    
                    strategy_id = py_file.stem
                    strategy_data = {
                        'name': strategy_id.replace('_', ' ').title(),
                        'code': code,
                        '_source': source,
                        '_file': str(py_file)
                    }
                    
                    # Try to extract metadata from code comments
                    name_match = re.search(r'class\s+(\w+)\s*\(', code)
                    if name_match:
                        strategy_data['name'] = name_match.group(1)
                    
                    strategies[strategy_id] = strategy_data
                    
                except Exception as e:
                    self.logger.warning(f"   Failed to load {py_file.name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading from {directory}: {e}")
        
        return strategies
    
    def _parse_all_strategies(self):
        """Parse all loaded strategies with AI"""
        self.logger.info("🧠 AI parsing all strategies...")
        
        for strategy_id, strategy_data in self.loaded_strategies.items():
            try:
                intelligence = self.intelligence_agent.parse_strategy(strategy_data)
                self.strategy_intelligence[strategy_id] = intelligence
            except Exception as e:
                self.logger.warning(f"Failed to parse {strategy_id}: {e}")
        
        self.logger.info(f"✅ Parsed {len(self.strategy_intelligence)} strategies with AI")
    
    def get_strategy_intelligence(self, strategy_id: str) -> Optional[StrategyIntelligence]:
        """Get AI intelligence for a strategy"""
        return self.strategy_intelligence.get(strategy_id)
    
    def get_all_intelligence(self) -> Dict[str, StrategyIntelligence]:
        """Get all strategy intelligence"""
        return self.strategy_intelligence

logger.info("✅ Strategy Loader Agent defined")

# =========================================================================================
# PAPER TRADING AGENT (SIMULATES TRADES WITH LEVERAGE, SL/TP)
# =========================================================================================

class PaperTradingAgent:
    """
    Simulates real trading with:
    - Leverage
    - Stop Loss / Take Profit
    - Position sizing
    - Fees and slippage
    - PnL tracking per strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.PAPER-TRADING")
        self.swarm = SwarmDecisionMaker()
        
        # Per-strategy state
        self.strategy_accounts: Dict[str, Dict] = {}
        self.strategy_positions: Dict[str, List[PaperPosition]] = {}
        self.trade_history: Dict[str, List[Dict]] = {}
        
    def initialize_strategy_account(self, strategy_id: str):
        """Initialize paper trading account for a strategy"""
        if strategy_id not in self.strategy_accounts:
            self.strategy_accounts[strategy_id] = {
                'capital': PaperTradingConfig.STARTING_CAPITAL_USD,
                'starting_capital': PaperTradingConfig.STARTING_CAPITAL_USD,
                'peak_capital': PaperTradingConfig.STARTING_CAPITAL_USD,
                'total_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_drawdown': 0.0,
                'last_updated': datetime.now()
            }
            self.strategy_positions[strategy_id] = []
            self.trade_history[strategy_id] = []
            
            self.logger.info(f"💰 Initialized account for {strategy_id}: ${PaperTradingConfig.STARTING_CAPITAL_USD:,.2f}")
    
    def process_signal(self, signal: TradingSignal, current_price: float):
        """
        Process a trading signal with SWARM CONSENSUS
        
        Args:
            signal: TradingSignal from AI
            current_price: Current market price
        """
        strategy_id = signal.strategy_id
        
        # Ensure account exists
        self.initialize_strategy_account(strategy_id)
        
        if signal.action == 'HOLD':
            return  # Nothing to do
        
        # Check if we have an existing position
        positions = self.strategy_positions.get(strategy_id, [])
        has_position = len([p for p in positions if p.status == 'OPEN']) > 0
        
        if signal.action == 'BUY':
            if has_position:
                # Check if we need to close a SHORT
                for pos in positions:
                    if pos.status == 'OPEN' and pos.direction == 'SHORT':
                        self._close_position(strategy_id, pos, current_price, 'Signal reversal')
                        break
                else:
                    self.logger.debug(f"{strategy_id}: Already have LONG position, skipping BUY")
                    return
            
            # Open new LONG position with SWARM approval
            self._open_position_with_swarm(strategy_id, signal, 'LONG', current_price)
            
        elif signal.action == 'SELL':
            if has_position:
                # Check if we need to close a LONG
                for pos in positions:
                    if pos.status == 'OPEN' and pos.direction == 'LONG':
                        self._close_position(strategy_id, pos, current_price, 'Signal reversal')
                        break
                else:
                    self.logger.debug(f"{strategy_id}: Already have SHORT position, skipping SELL")
                    return
            
            # Open new SHORT position with SWARM approval
            self._open_position_with_swarm(strategy_id, signal, 'SHORT', current_price)
    
    def _open_position_with_swarm(self, strategy_id: str, signal: TradingSignal,
                                   direction: str, current_price: float):
        """Open position after getting SWARM consensus"""
        
        account = self.strategy_accounts[strategy_id]
        
        # Calculate position size (30% of capital with leverage)
        base_size = account['capital'] * PaperTradingConfig.MAX_POSITION_PERCENT
        leveraged_size = base_size * PaperTradingConfig.DEFAULT_LEVERAGE
        
        # Get SWARM consensus
        self.logger.info("=" * 60)
        self.logger.info(f"🎯 SIGNAL: {signal.strategy_name}")
        self.logger.info(f"   Direction: {direction}")
        self.logger.info(f"   Symbol: {signal.symbol}")
        self.logger.info(f"   Price: ${current_price:.2f}")
        self.logger.info(f"   Reason: {signal.reason}")
        self.logger.info("=" * 60)
        
        consensus = self.swarm.get_swarm_consensus(
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            direction=direction,
            size_usd=leveraged_size,
            reason=signal.reason,
            indicators=signal.indicators,
            current_price=current_price
        )
        
        if consensus['decision'] != 'APPROVE':
            self.logger.warning(f"❌ SWARM REJECTED: {consensus['reason']}")
            self.logger.warning(f"   Votes: {consensus['votes']}")
            return
        
        self.logger.info(f"✅ SWARM APPROVED: {consensus['reason']}")
        self.logger.info(f"   Votes: {consensus['votes']}")
        self.logger.info(f"   Confidence: {consensus['confidence']:.0%}")
        
        # Use recommended size
        final_size = consensus['recommended_size']
        
        # Calculate fees
        fees = final_size * PaperTradingConfig.TRADING_FEE_PERCENT
        
        # Apply slippage
        if direction == 'LONG':
            entry_price = current_price * (1 + PaperTradingConfig.SLIPPAGE_PERCENT)
        else:
            entry_price = current_price * (1 - PaperTradingConfig.SLIPPAGE_PERCENT)
        
        # Calculate SL/TP prices
        if direction == 'LONG':
            stop_loss_price = entry_price * (1 - PaperTradingConfig.DEFAULT_STOP_LOSS_PERCENT)
            take_profit_price = entry_price * (1 + PaperTradingConfig.DEFAULT_TAKE_PROFIT_PERCENT)
        else:
            stop_loss_price = entry_price * (1 + PaperTradingConfig.DEFAULT_STOP_LOSS_PERCENT)
            take_profit_price = entry_price * (1 - PaperTradingConfig.DEFAULT_TAKE_PROFIT_PERCENT)
        
        # Create position
        position = PaperPosition(
            strategy_id=strategy_id,
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_price,
            size_usd=final_size,
            size_coins=final_size / entry_price,
            leverage=PaperTradingConfig.DEFAULT_LEVERAGE,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            entry_time=datetime.now(),
            fees_paid=fees
        )
        
        # Deduct margin from capital
        margin = final_size / PaperTradingConfig.DEFAULT_LEVERAGE
        account['capital'] -= (margin + fees)
        
        # Add to positions
        self.strategy_positions[strategy_id].append(position)
        
        # Log trade
        self._log_trade_open(strategy_id, position, signal)
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 POSITION OPENED (Paper)")
        self.logger.info(f"   Strategy: {signal.strategy_name}")
        self.logger.info(f"   Direction: {direction}")
        self.logger.info(f"   Size: ${final_size:.2f} ({PaperTradingConfig.DEFAULT_LEVERAGE}x leverage)")
        self.logger.info(f"   Entry: ${entry_price:.2f}")
        self.logger.info(f"   Stop Loss: ${stop_loss_price:.2f}")
        self.logger.info(f"   Take Profit: ${take_profit_price:.2f}")
        self.logger.info(f"   Fees: ${fees:.2f}")
        self.logger.info("=" * 60)
    
    def _close_position(self, strategy_id: str, position: PaperPosition,
                        current_price: float, reason: str):
        """Close a paper position"""
        
        account = self.strategy_accounts[strategy_id]
        
        # Apply slippage
        if position.direction == 'LONG':
            exit_price = current_price * (1 - PaperTradingConfig.SLIPPAGE_PERCENT)
        else:
            exit_price = current_price * (1 + PaperTradingConfig.SLIPPAGE_PERCENT)
        
        # Calculate PnL
        if position.direction == 'LONG':
            pnl_percent = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_percent = (position.entry_price - exit_price) / position.entry_price
        
        pnl_usd = position.size_usd * pnl_percent
        
        # Exit fees
        exit_fees = position.size_usd * PaperTradingConfig.TRADING_FEE_PERCENT
        total_fees = position.fees_paid + exit_fees
        
        # Net PnL
        net_pnl = pnl_usd - exit_fees
        
        # Update position
        position.status = 'CLOSED'
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.pnl_usd = net_pnl
        position.pnl_percent = pnl_percent * 100
        position.fees_paid = total_fees
        
        # Return margin + PnL to account
        margin = position.size_usd / PaperTradingConfig.DEFAULT_LEVERAGE
        account['capital'] += margin + net_pnl
        account['total_pnl'] += net_pnl
        account['total_trades'] += 1
        
        if net_pnl > 0:
            account['winning_trades'] += 1
        else:
            account['losing_trades'] += 1
        
        # Update peak and drawdown
        if account['capital'] > account['peak_capital']:
            account['peak_capital'] = account['capital']
        
        current_drawdown = (account['peak_capital'] - account['capital']) / account['peak_capital']
        if current_drawdown > account['max_drawdown']:
            account['max_drawdown'] = current_drawdown
        
        account['last_updated'] = datetime.now()
        
        # Log trade close
        self._log_trade_close(strategy_id, position, reason)
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 POSITION CLOSED (Paper)")
        self.logger.info(f"   Strategy: {strategy_id}")
        self.logger.info(f"   Direction: {position.direction}")
        self.logger.info(f"   Entry: ${position.entry_price:.2f}")
        self.logger.info(f"   Exit: ${exit_price:.2f}")
        self.logger.info(f"   PnL: ${net_pnl:.2f} ({position.pnl_percent:.2f}%)")
        self.logger.info(f"   Reason: {reason}")
        self.logger.info(f"   Account Capital: ${account['capital']:.2f}")
        self.logger.info("=" * 60)
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]):
        """Check all open positions for SL/TP hits"""
        
        for strategy_id, positions in self.strategy_positions.items():
            for position in positions:
                if position.status != 'OPEN':
                    continue
                
                current_price = current_prices.get(position.symbol)
                if current_price is None:
                    continue
                
                # Check stop loss
                if position.direction == 'LONG':
                    if current_price <= position.stop_loss_price:
                        position.status = 'STOPPED'
                        self._close_position(strategy_id, position, current_price, 'STOP LOSS HIT')
                    elif current_price >= position.take_profit_price:
                        position.status = 'TP_HIT'
                        self._close_position(strategy_id, position, current_price, 'TAKE PROFIT HIT')
                else:  # SHORT
                    if current_price >= position.stop_loss_price:
                        position.status = 'STOPPED'
                        self._close_position(strategy_id, position, current_price, 'STOP LOSS HIT')
                    elif current_price <= position.take_profit_price:
                        position.status = 'TP_HIT'
                        self._close_position(strategy_id, position, current_price, 'TAKE PROFIT HIT')
    
    def _log_trade_open(self, strategy_id: str, position: PaperPosition, signal: TradingSignal):
        """Log trade open to file"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy_id': strategy_id,
            'strategy_name': signal.strategy_name,
            'action': 'OPEN',
            'direction': position.direction,
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'size_usd': position.size_usd,
            'leverage': position.leverage,
            'stop_loss': position.stop_loss_price,
            'take_profit': position.take_profit_price,
            'reason': signal.reason,
            'indicators': signal.indicators
        }
        
        self.trade_history[strategy_id].append(trade_record)
        self._save_trade_to_file(trade_record)
    
    def _log_trade_close(self, strategy_id: str, position: PaperPosition, reason: str):
        """Log trade close to file"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy_id': strategy_id,
            'action': 'CLOSE',
            'direction': position.direction,
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'size_usd': position.size_usd,
            'pnl_usd': position.pnl_usd,
            'pnl_percent': position.pnl_percent,
            'fees_paid': position.fees_paid,
            'reason': reason,
            'duration_minutes': (position.exit_time - position.entry_time).total_seconds() / 60
        }
        
        self.trade_history[strategy_id].append(trade_record)
        self._save_trade_to_file(trade_record)
    
    def _save_trade_to_file(self, trade_record: Dict):
        """Save trade record to JSONL file"""
        try:
            trade_file = PaperTradingConfig.PAPER_TRADES_DIR / "all_trades.jsonl"
            with open(trade_file, 'a') as f:
                f.write(json.dumps(trade_record, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
    
    def get_strategy_performance(self, strategy_id: str) -> Dict:
        """Get performance metrics for a strategy"""
        account = self.strategy_accounts.get(strategy_id, {})
        if not account:
            return {}
        
        total_trades = account.get('total_trades', 0)
        winning_trades = account.get('winning_trades', 0)
        
        return {
            'strategy_id': strategy_id,
            'starting_capital': account.get('starting_capital', 0),
            'current_capital': account.get('capital', 0),
            'total_pnl': account.get('total_pnl', 0),
            'pnl_percent': (account.get('total_pnl', 0) / account.get('starting_capital', 1)) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': account.get('losing_trades', 0),
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'max_drawdown': account.get('max_drawdown', 0) * 100,
            'open_positions': len([p for p in self.strategy_positions.get(strategy_id, []) if p.status == 'OPEN'])
        }
    
    def get_all_performance(self) -> List[Dict]:
        """Get performance for all strategies"""
        return [self.get_strategy_performance(sid) for sid in self.strategy_accounts.keys()]

logger.info("✅ Paper Trading Agent defined")

# =========================================================================================
# PERFORMANCE LOGGER (LOGS ALL RESULTS)
# =========================================================================================

class PerformanceLogger:
    """
    Logs performance metrics for all strategies
    Creates reports to see which strategies would profit in real life
    """
    
    def __init__(self, paper_trading_agent: PaperTradingAgent):
        self.logger = logging.getLogger("TRADEPEXTEST.PERFORMANCE")
        self.trading_agent = paper_trading_agent
        self.last_log_time = datetime.now()
        
    def log_performance_snapshot(self):
        """Log current performance for all strategies"""
        
        performances = self.trading_agent.get_all_performance()
        
        if not performances:
            return
        
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info("📊 STRATEGY PERFORMANCE SNAPSHOT")
        self.logger.info("=" * 100)
        
        # Sort by PnL
        performances.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
        
        # Header
        self.logger.info(f"{'Strategy':<40} {'Capital':>12} {'PnL':>12} {'Win Rate':>10} {'Trades':>8} {'DD':>8}")
        self.logger.info("-" * 100)
        
        for perf in performances:
            strategy_name = perf['strategy_id'][:38]
            capital = f"${perf['current_capital']:,.0f}"
            pnl = f"${perf['total_pnl']:+,.0f}"
            win_rate = f"{perf['win_rate']:.1f}%"
            trades = str(perf['total_trades'])
            drawdown = f"{perf['max_drawdown']:.1f}%"
            
            # Color coding (in log)
            if perf['total_pnl'] > 0:
                self.logger.info(f"✅ {strategy_name:<38} {capital:>12} {pnl:>12} {win_rate:>10} {trades:>8} {drawdown:>8}")
            else:
                self.logger.info(f"❌ {strategy_name:<38} {capital:>12} {pnl:>12} {win_rate:>10} {trades:>8} {drawdown:>8}")
        
        self.logger.info("=" * 100)
        
        # Summary statistics
        total_strategies = len(performances)
        profitable = len([p for p in performances if p['total_pnl'] > 0])
        total_pnl = sum(p['total_pnl'] for p in performances)
        
        self.logger.info(f"📈 Total Strategies: {total_strategies}")
        self.logger.info(f"✅ Profitable: {profitable} ({profitable/total_strategies*100:.1f}%)" if total_strategies > 0 else "")
        self.logger.info(f"💰 Combined PnL: ${total_pnl:+,.2f}")
        self.logger.info("")
        
        # Save to file
        self._save_performance_report(performances)
    
    def _save_performance_report(self, performances: List[Dict]):
        """Save performance report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = PaperTradingConfig.STRATEGY_PERFORMANCE_DIR / f"performance_{timestamp}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategies': performances,
                'summary': {
                    'total_strategies': len(performances),
                    'profitable': len([p for p in performances if p['total_pnl'] > 0]),
                    'total_pnl': sum(p['total_pnl'] for p in performances)
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also update a "latest" file
            latest_file = PaperTradingConfig.STRATEGY_PERFORMANCE_DIR / "latest_performance.json"
            with open(latest_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
    
    def generate_final_report(self) -> Dict:
        """Generate final comprehensive report"""
        
        performances = self.trading_agent.get_all_performance()
        
        # Sort by profitability
        performances.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
        
        report = {
            'report_time': datetime.now().isoformat(),
            'total_strategies_tested': len(performances),
            'profitable_strategies': [],
            'unprofitable_strategies': [],
            'top_5_strategies': [],
            'worst_5_strategies': [],
            'overall_statistics': {}
        }
        
        for perf in performances:
            if perf['total_pnl'] > 0:
                report['profitable_strategies'].append(perf)
            else:
                report['unprofitable_strategies'].append(perf)
        
        report['top_5_strategies'] = performances[:5]
        report['worst_5_strategies'] = performances[-5:] if len(performances) >= 5 else performances
        
        total_pnl = sum(p['total_pnl'] for p in performances)
        total_trades = sum(p['total_trades'] for p in performances)
        total_wins = sum(p['winning_trades'] for p in performances)
        
        report['overall_statistics'] = {
            'total_combined_pnl': total_pnl,
            'average_pnl_per_strategy': total_pnl / len(performances) if performances else 0,
            'total_trades': total_trades,
            'overall_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'profitable_strategy_count': len(report['profitable_strategies']),
            'unprofitable_strategy_count': len(report['unprofitable_strategies'])
        }
        
        # Save final report
        final_file = PaperTradingConfig.RESULTS_DIR / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Final report saved to: {final_file}")
        
        return report

logger.info("✅ Performance Logger defined")

# =========================================================================================
# MAIN TRADING ENGINE (24/7 CONTINUOUS OPERATION)
# =========================================================================================

class TradePexTestEngine:
    """
    Main engine that runs 24/7 paper trading on all strategies
    
    Features:
    - Loads all strategies dynamically
    - Uses AI to understand each strategy
    - Fetches real market data
    - Calculates required indicators per strategy
    - Generates trading signals with AI
    - Executes paper trades with SWARM consensus
    - Logs all results for analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TRADEPEXTEST.SYSTEM")
        
        # Initialize all components
        self.strategy_loader = StrategyLoaderAgent()
        self.paper_trader = PaperTradingAgent()
        self.performance_logger = PerformanceLogger(self.paper_trader)
        
        # State
        self.running = False
        self.iteration = 0
        self.start_time = None
        
    def run(self):
        """Main 24/7 trading loop"""
        
        self.running = True
        self.start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 TRADEPEXTEST ENGINE STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"   Mode: 24/7 Paper Trading")
        self.logger.info(f"   Capital per strategy: ${PaperTradingConfig.STARTING_CAPITAL_USD:,.2f}")
        self.logger.info(f"   Leverage: {PaperTradingConfig.DEFAULT_LEVERAGE}x")
        self.logger.info(f"   Stop Loss: {PaperTradingConfig.DEFAULT_STOP_LOSS_PERCENT*100:.0f}%")
        self.logger.info(f"   Take Profit: {PaperTradingConfig.DEFAULT_TAKE_PROFIT_PERCENT*100:.0f}%")
        self.logger.info("=" * 80)
        
        # Phase 1: Load all strategies
        self.logger.info("")
        self.logger.info("📂 PHASE 1: Loading all strategies...")
        strategies = self.strategy_loader.load_all_strategies()
        
        if not strategies:
            self.logger.error("❌ No strategies found! Please add strategies to strategy_library/")
            return
        
        self.logger.info(f"✅ Loaded {len(strategies)} strategies")
        
        # Phase 2: Get AI intelligence for each strategy
        self.logger.info("")
        self.logger.info("🧠 PHASE 2: AI analyzing all strategies...")
        all_intelligence = self.strategy_loader.get_all_intelligence()
        
        for strategy_id, intelligence in all_intelligence.items():
            self.logger.info(f"   {strategy_id}:")
            self.logger.info(f"      Type: {intelligence.strategy_type}")
            self.logger.info(f"      Indicators: {intelligence.required_indicators}")
            self.logger.info(f"      Coins: {intelligence.preferred_coins}")
        
        # Phase 3: Initialize paper trading accounts
        self.logger.info("")
        self.logger.info("💰 PHASE 3: Initializing paper trading accounts...")
        for strategy_id in strategies.keys():
            self.paper_trader.initialize_strategy_account(strategy_id)
        
        # Phase 4: Start main trading loop
        self.logger.info("")
        self.logger.info("🔄 PHASE 4: Starting 24/7 trading loop...")
        self.logger.info(f"   Check interval: {PaperTradingConfig.STRATEGY_CHECK_INTERVAL_SECONDS}s")
        self.logger.info("")
        
        last_performance_log = time.time()
        
        try:
            while self.running:
                self.iteration += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Trading cycle #{self.iteration}")
                
                # Get current prices for all coins
                current_prices = self._fetch_current_prices()
                
                if not current_prices:
                    self.logger.warning("⚠️ Could not fetch prices, skipping cycle")
                    time.sleep(30)
                    continue
                
                # Check SL/TP for all open positions
                self.paper_trader.check_stop_loss_take_profit(current_prices)
                
                # Process each strategy
                for strategy_id, intelligence in all_intelligence.items():
                    try:
                        self._process_strategy(strategy_id, intelligence, current_prices)
                    except Exception as e:
                        self.logger.error(f"Error processing {strategy_id}: {e}")
                        continue
                
                # Log performance periodically
                if time.time() - last_performance_log > PaperTradingConfig.PERFORMANCE_LOG_INTERVAL_SECONDS:
                    self.performance_logger.log_performance_snapshot()
                    last_performance_log = time.time()
                
                # Calculate cycle time
                cycle_duration = time.time() - cycle_start
                self.logger.debug(f"   Cycle completed in {cycle_duration:.2f}s")
                
                # Sleep until next cycle
                sleep_time = max(0, PaperTradingConfig.STRATEGY_CHECK_INTERVAL_SECONDS - cycle_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("")
            self.logger.info("🛑 Shutdown requested...")
            self._shutdown()
    
    def _fetch_current_prices(self) -> Dict[str, float]:
        """Fetch current prices for all tradeable coins"""
        prices = {}
        
        for coin in PaperTradingConfig.TRADEABLE_COINS:
            symbol = f"{coin.lower()}usdt"
            price = market_data_fetcher.get_current_price(symbol)
            if price:
                prices[coin] = price
        
        return prices
    
    def _process_strategy(self, strategy_id: str, intelligence: StrategyIntelligence,
                          current_prices: Dict[str, float]):
        """Process a single strategy for trading signals"""
        
        # Determine which coin to trade
        preferred_coin = intelligence.preferred_coins[0] if intelligence.preferred_coins else 'BTC'
        
        if preferred_coin not in current_prices:
            return
        
        current_price = current_prices[preferred_coin]
        
        # Fetch market data for the coin
        symbol = f"{preferred_coin.lower()}usdt"
        market_data = market_data_fetcher.fetch_candles(symbol, '15min', 500)
        
        if market_data is None or len(market_data) < 50:
            return
        
        # Calculate required indicators
        indicators = indicator_engine.calculate_all_indicators(
            market_data, 
            intelligence.required_indicators
        )
        
        if not indicators:
            return
        
        # Check if we have an open position
        positions = self.paper_trader.strategy_positions.get(strategy_id, [])
        open_positions = [p for p in positions if p.status == 'OPEN']
        has_position = len(open_positions) > 0
        current_direction = open_positions[0].direction if has_position else None
        
        # Generate trading signal using AI
        signal = signal_generator.generate_signal(
            intelligence,
            indicators,
            has_position,
            current_direction
        )
        
        # Process signal
        if signal.action != 'HOLD':
            self.paper_trader.process_signal(signal, current_price)
    
    def _shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        # Generate final report
        self.logger.info("")
        self.logger.info("📊 Generating final performance report...")
        report = self.performance_logger.generate_final_report()
        
        # Print summary
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 FINAL RESULTS SUMMARY")
        self.logger.info("=" * 80)
        
        stats = report['overall_statistics']
        self.logger.info(f"   Total Strategies Tested: {report['total_strategies_tested']}")
        self.logger.info(f"   Profitable Strategies: {stats['profitable_strategy_count']}")
        self.logger.info(f"   Unprofitable Strategies: {stats['unprofitable_strategy_count']}")
        self.logger.info(f"   Total Combined PnL: ${stats['total_combined_pnl']:+,.2f}")
        self.logger.info(f"   Total Trades: {stats['total_trades']}")
        self.logger.info(f"   Overall Win Rate: {stats['overall_win_rate']:.1f}%")
        
        if report['top_5_strategies']:
            self.logger.info("")
            self.logger.info("🏆 TOP 5 STRATEGIES:")
            for i, strat in enumerate(report['top_5_strategies'][:5], 1):
                self.logger.info(f"   {i}. {strat['strategy_id']}: ${strat['total_pnl']:+,.2f} ({strat['win_rate']:.1f}% win rate)")
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("👋 TradePexTest shutdown complete")
        self.logger.info("=" * 80)

# =========================================================================================
# ENTRY POINT
# =========================================================================================

def print_startup_banner():
    """Print startup banner"""
    cprint("=" * 80, "cyan")
    cprint("🚀 TRADEPEXTEST - 24/7 INTELLIGENT PAPER TRADING TESTER", "cyan", attrs=['bold'])
    cprint("=" * 80, "cyan")
    cprint("", "cyan")
    cprint("   Version: 1.0 (FULL IMPLEMENTATION)", "yellow")
    cprint("   Mode: PAPER TRADING - NO REAL MONEY", "green")
    cprint("   AI Intelligence: ENABLED", "magenta")
    cprint("   SWARM Consensus: ENABLED (Multi-LLM voting)", "magenta")
    cprint("", "cyan")
    cprint(f"   Starting Capital: ${PaperTradingConfig.STARTING_CAPITAL_USD:,.2f} per strategy", "white")
    cprint(f"   Leverage: {PaperTradingConfig.DEFAULT_LEVERAGE}x", "white")
    cprint(f"   Stop Loss: {PaperTradingConfig.DEFAULT_STOP_LOSS_PERCENT*100:.0f}%", "white")
    cprint(f"   Take Profit: {PaperTradingConfig.DEFAULT_TAKE_PROFIT_PERCENT*100:.0f}%", "white")
    cprint("", "cyan")
    cprint("   Features:", "yellow")
    cprint("   1. 🧠 AI Strategy Parser - Understands each strategy dynamically", "white")
    cprint("   2. 📊 Dynamic Indicators - Calculates what each strategy needs", "white")
    cprint("   3. 🤖 AI Signal Generator - Interprets entry/exit rules", "white")
    cprint("   4. 🗳️ SWARM Consensus - Multiple LLMs vote on trades", "white")
    cprint("   5. 📈 Real Data - Uses HTX exchange for real candles", "white")
    cprint("   6. 💰 Full Simulation - Leverage, SL/TP, fees included", "white")
    cprint("   7. 📋 Performance Logs - See which strategies profit", "white")
    cprint("", "cyan")
    cprint("=" * 80, "cyan")
    cprint("", "cyan")

def main():
    """Main entry point"""
    
    print_startup_banner()
    
    # Verify API keys
    available_model = ModelFactory.get_available_model()
    if available_model:
        cprint(f"✅ LLM Available: {available_model['type']}/{available_model['name']}", "green")
    else:
        cprint("⚠️ No LLM API key found - using rule-based fallback", "yellow")
    
    cprint("", "cyan")
    
    # Create and run engine
    engine = TradePexTestEngine()
    
    try:
        engine.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

# =========================================================================================
# END OF TRADEPEXTEST.PY
# =========================================================================================

logger.info("=" * 80)
logger.info("🎉 TRADEPEXTEST SYSTEM - COMPLETE IMPLEMENTATION LOADED")
logger.info("=" * 80)
logger.info(f"Total: 2300+ lines of REAL, FUNCTIONAL CODE")
logger.info(f"NO PLACEHOLDERS - NO SIMPLIFIED CODE")
logger.info(f"")
logger.info(f"Based on:")
logger.info(f"  - TradePex (2299 lines)")
logger.info(f"  - AI Strategy Intelligence Layer")
logger.info(f"  - SWARM Multi-LLM Consensus")
logger.info(f"  - Dynamic Indicator Engine")
logger.info(f"  - Real HTX Market Data")
logger.info(f"  - Full Paper Trading Simulation")
logger.info(f"")
logger.info(f"Ready to launch with: python tradepextest.py")
logger.info("=" * 80)
