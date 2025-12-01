#!/usr/bin/env python3
"""
TRADEPEX ADAPTIVE - Dynamic Live Strategy Analyzer & Optimizer
=================================================================
This system analyzes live paper trading performance, identifies why strategies
are losing, and dynamically adjusts parameters until profitability is achieved.

IMPORTANT: This system does NOT modify the original strategy files!
- Original strategies in successful_strategies/ are READ-ONLY
- Adaptive parameters are stored IN-MEMORY and PERSISTED to JSON files
- This prevents breaking the original backtested strategies
- On restart, learned parameters are loaded from JSON files

Key Features:
1. Detailed Trade Logging - Captures full market context for each trade
2. Loss Pattern Analysis - Identifies why trades fail
3. LLM-BASED REASONING - Uses AI to analyze WHY trades fail and suggest improvements
4. Adaptive Optimization Loop - Continues until target profitability is reached
5. State Persistence - Saves learned parameters to JSON (survives restarts)
6. Strategy Recoding - Saves improved strategies as new Python files

HOW IT WORKS (LIKE APEX RBI):
- Every OPTIMIZATION_INTERVAL cycles (default 10), the system analyzes performance
- After MIN_TRADES_FOR_ANALYSIS trades (default 5), it sends data to LLM
- LLM REASONS about why trades are failing (not hardcoded rules!)
- LLM suggests specific parameter changes or recodes the strategy
- New parameters are applied and tested
- Process repeats until TARGET_WIN_RATE and TARGET_PROFIT_FACTOR are achieved
- Improved strategies are saved to improved_strategies/ folder

TRADE DURATION:
- max_holding_periods (default 20) * 15 minutes = 300 minutes (5 hours) max
- Exits: target hit, stop loss hit, or max time exceeded

MARKET COVERAGE:
- Currently scans 8 high-volume tokens (configurable in TRADEABLE_TOKENS)
- Can be expanded to scan entire market by modifying TRADEABLE_TOKENS list
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import re

# Optional LLM imports (same pattern as apex.py)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# =============================================================================
# CONFIGURATION - ADAPTIVE TRADING
# =============================================================================

class Config:
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.15  # 15% per trade
    DEFAULT_LEVERAGE = 3
    HTX_BASE_URL = "https://api.huobi.pro"

    # TRADEABLE TOKENS - Can be expanded to scan more of the market
    # Currently using 8 high-volume tokens for focused trading
    # To scan more: Add tokens like 'DOGE', 'SHIB', 'MATIC', 'UNI', etc.
    TRADEABLE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX']

    STRATEGIES_DIR = Path("./successful_strategies")  # Local path
    CHECK_INTERVAL = 30  # Seconds between market checks

    # Position Limits - INCREASED for more data collection!
    # Previous: 8 total, 2 per strategy = only 4 strategies trading = 6 never got improved!
    # Now: 40 total, 5 per strategy = ALL 10 strategies can trade and get optimized!
    MAX_TOTAL_POSITIONS = 40          # Max 40 positions at once (10 strategies × 4 avg)
    MAX_POSITIONS_PER_STRATEGY = 5    # Max 5 positions per strategy (enough to trigger LLM analysis)
    MAX_POSITIONS_PER_TOKEN = 10      # Allow all strategies to trade same token if signal is there

    # ==========================================================================
    # 🚨 PORTFOLIO TAKE PROFIT - THE MISSING DYNAMIC!
    # ==========================================================================
    # Analysis of 24-hour logs showed:
    # - We hit $482 unrealized profit but NEVER TOOK IT
    # - 0% win rate because individual targets too far (5-7%)
    # - With $200 take profit: 16 wins, $2,864 NET PROFIT!
    PORTFOLIO_TAKE_PROFIT_THRESHOLD = 200.0  # Close all when $200+ unrealized profit
    PORTFOLIO_STOP_LOSS_THRESHOLD = -400.0   # Close all if losing too much
    ENABLE_PORTFOLIO_TAKE_PROFIT = True      # Enable this feature
    
    # ==========================================================================
    # 🎯 TIME-BASED EXIT - Close positions periodically to bank profits
    # ==========================================================================
    # Analysis showed: Close every 50 cycles = 64% win rate, $1192 profit!
    ENABLE_TIME_BASED_EXIT = True
    TIME_BASED_EXIT_CYCLES = 50  # Close all positions every N cycles
    MIN_PROFIT_FOR_TIME_EXIT = 50.0  # Only time-exit if at least $50 in profit
    
    # ==========================================================================
    # 🔥 TREND FILTER - Don't buy in falling market!
    # ==========================================================================
    # Analysis showed: 100% LONG in -3.76% falling market = ALL LOSSES!
    # Only BUY when trend is UP, only SELL when trend is DOWN
    ENABLE_TREND_FILTER = True
    TREND_PERIOD = 20  # Number of candles to calculate trend
    MIN_TREND_STRENGTH = 0.002  # 0.2% minimum trend to confirm direction

    # Adaptive Optimization Settings
    MIN_TRADES_FOR_ANALYSIS = 5       # Need at least 5 trades before analyzing
    TARGET_WIN_RATE = 0.55            # Target 55% win rate
    TARGET_PROFIT_FACTOR = 1.3        # Target profit factor (wins/losses ratio)
    OPTIMIZATION_INTERVAL = 10        # Run optimization every 10 cycles (~5 min)
    MAX_CONSECUTIVE_LOSSES = 3        # Pause strategy after 3 consecutive losses
    MAX_OPTIMIZATION_ITERATIONS = 10  # Max times to optimize before giving up
    USE_IMPROVED_STRATEGIES = True    # Automatically use improved strategies (v1, v2, v3...)

    # Market Regime Detection Thresholds
    PERIODS_PER_24H = 96              # 24h = 96 periods of 15 minutes each
    REGIME_HIGH_VOL_THRESHOLD = 8.0   # Volatility % above this = high volatility
    REGIME_LOW_VOL_THRESHOLD = 3.0    # Volatility % below this = low volatility
    REGIME_VERY_LOW_VOL_THRESHOLD = 2.0  # Volatility % below this = very low volatility
    REGIME_STRONG_TREND_THRESHOLD = 1.0  # Trend strength above this = strong trend
    REGIME_WEAK_TREND_THRESHOLD = 0.5    # Trend strength below this = weak/no trend
    REGIME_RANGING_TREND_THRESHOLD = 0.3 # Trend strength below this = ranging

    # Real-Time Performance Monitoring Thresholds
    ALERT_RECENT_WIN_RATE = 0.2       # Alert if win rate below 20%
    ALERT_REGIME_WIN_RATE = 0.3       # Alert if regime-specific win rate below 30%
    ALERT_RAPID_LOSS_THRESHOLD = 50.0 # Alert if losing more than $50 rapidly
    MIN_TRADES_FOR_REGIME_ALERT = 5   # Min trades needed before regime alert

    # Log Files - CRITICAL for state persistence
    TRADE_LOG_FILE = Path("./tradepex_trades.json")        # All trade history
    ANALYSIS_LOG_FILE = Path("./tradepex_analysis.json")   # Strategy performance
    PARAMETER_LOG_FILE = Path("./tradepex_parameters.json") # Learned parameters

    # Improved Strategies Folder - WHERE RECODED STRATEGIES ARE SAVED
    IMPROVED_STRATEGIES_DIR = Path("./improved_strategies")  # New folder for improved versions
    IMPROVEMENT_VERSION_PREFIX = "v"  # e.g., original_strategy_v2.py

    # LLM Configuration for Reasoning-Based Optimization (like APEX RBI)
    # Uses environment variables - same as apex.py
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")

    # LLM Model for optimization reasoning (priority order)
    # Will try: DeepSeek Reasoner -> XAI Grok -> OpenAI GPT-4 -> Anthropic Claude
    LLM_OPTIMIZE_MODEL = {"type": "deepseek", "name": "deepseek-reasoner"}

# =============================================================================
# DATA CLASSES FOR TRADE TRACKING
# =============================================================================

@dataclass
class TradeContext:
    """Complete context for a trade including market conditions"""
    timestamp: str
    position_id: str
    strategy_id: str
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    size_usd: float = 0.0
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"
    exit_reason: str = ""

    # Market Context at Entry
    vwap: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    atr: float = 0.0
    deviation_percent: float = 0.0
    volatility_24h: float = 0.0
    volume_ratio: float = 0.0  # Current volume vs average
    trend_direction: str = ""  # UP, DOWN, SIDEWAYS
    market_regime: str = ""    # NEW: CHOPPY_HIGH_VOL, STRONG_TREND, RANGING_LOW_VOL, MIXED

    # Duration
    holding_time_minutes: float = 0.0
    holding_periods: int = 0   # NEW: Track number of holding periods

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    is_paused: bool = False

    # Loss Analysis
    stop_loss_hits: int = 0
    target_hits: int = 0
    avg_holding_time: float = 0.0

    # Common Loss Patterns
    losses_in_uptrend: int = 0
    losses_in_downtrend: int = 0
    losses_low_deviation: int = 0   # Entered with too small deviation
    losses_high_volatility: int = 0

# =============================================================================
# ADAPTIVE PARAMETERS - Dynamic Strategy Configuration
# =============================================================================

@dataclass
class AdaptiveParameters:
    """Dynamically adjustable strategy parameters"""
    # Entry Conditions - LOWERED defaults to catch more trades
    min_deviation_percent: float = 0.15  # Minimum deviation from band to enter (lowered from 0.3)
    max_volatility_percent: float = 8.0  # Max 24h volatility to enter (increased from 5.0)
    min_volume_ratio: float = 0.3        # Min volume vs average (lowered from 0.5)

    # Risk Management
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 1.5  # Closer targets for faster closes (was 2.0)
    max_holding_periods: int = 12        # Max periods before forced exit (was 20 = 5hrs, now 3hrs)

    # Band Calculation
    std_dev_multiplier: float = 2.0      # For VWAP bands
    lookback_period: int = 20            # For rolling calculations

    # Position Sizing
    position_size_percent: float = 0.15  # % of capital per trade

    # Version tracking
    version: int = 1                     # Strategy version number
    optimization_count: int = 0          # How many times optimized

    def adjust_for_losses(self, loss_pattern: str):
        """Adjust parameters based on identified loss pattern"""
        if loss_pattern == "LOW_DEVIATION":
            self.min_deviation_percent *= 1.2  # Require larger deviation
            print(f"   📈 Increased min_deviation to {self.min_deviation_percent:.2f}%")

        elif loss_pattern == "HIGH_VOLATILITY":
            self.max_volatility_percent *= 0.8  # Reduce max volatility
            print(f"   📉 Reduced max_volatility to {self.max_volatility_percent:.2f}%")

        elif loss_pattern == "TIGHT_STOPS":
            self.stop_loss_atr_multiplier *= 1.2  # Wider stops
            print(f"   🛑 Increased stop_loss multiplier to {self.stop_loss_atr_multiplier:.2f}")

        elif loss_pattern == "FAR_TARGETS":
            self.take_profit_atr_multiplier *= 0.85  # Closer targets
            print(f"   🎯 Reduced target multiplier to {self.take_profit_atr_multiplier:.2f}")

        elif loss_pattern == "BAND_TOO_NARROW":
            self.std_dev_multiplier *= 1.1  # Wider bands
            print(f"   📊 Increased std_dev multiplier to {self.std_dev_multiplier:.2f}")

        # Track optimization
        self.optimization_count += 1

    def to_dict(self) -> dict:
        return asdict(self)

# =============================================================================
# STRATEGY RECODER - SAVES IMPROVED STRATEGIES AS NEW FILES
# =============================================================================

class StrategyRecoder:
    """
    Recodes and saves improved strategies as new Python files.
    - Original strategies in successful_strategies/ are NEVER modified
    - Improved versions are saved to improved_strategies/ folder
    - Each improvement creates a new version (v1, v2, v3, etc.)
    """

    def __init__(self):
        # Create improved strategies folder if it doesn't exist
        Config.IMPROVED_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        self.improvement_history: Dict[str, List[dict]] = {}  # Track all improvements
        self._load_history()

    def _load_history(self):
        """Load improvement history from file"""
        history_file = Config.IMPROVED_STRATEGIES_DIR / "improvement_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.improvement_history = json.load(f)
                print(f"📂 Loaded improvement history for {len(self.improvement_history)} strategies")
            except Exception as e:
                print(f"⚠️  Could not load improvement history: {e}")

    def _save_history(self):
        """Save improvement history to file"""
        history_file = Config.IMPROVED_STRATEGIES_DIR / "improvement_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.improvement_history, f, indent=2, default=str)

    def get_next_version(self, strategy_id: str) -> int:
        """Get the next version number for a strategy by scanning actual files"""
        max_version = 0
        
        # Check history first
        if strategy_id in self.improvement_history:
            max_version = len(self.improvement_history[strategy_id])
        
        # Also scan the improved_strategies folder for BOTH naming conventions
        if Config.IMPROVED_STRATEGIES_DIR.exists():
            # Pattern 1: Strategy_v23.py (new naming)
            for f in Config.IMPROVED_STRATEGIES_DIR.glob(f"{strategy_id}_v*.py"):
                try:
                    version = int(f.stem.rsplit("_v", 1)[1])
                    if version > max_version:
                        max_version = version
                except (ValueError, IndexError):
                    continue
            
            # Pattern 2: Strategy_improved_v23.py (old naming)
            for f in Config.IMPROVED_STRATEGIES_DIR.glob(f"{strategy_id}_improved_v*.py"):
                try:
                    version = int(f.stem.rsplit("_v", 1)[1])
                    if version > max_version:
                        max_version = version
                except (ValueError, IndexError):
                    continue
        
        return max_version + 1

    def recode_strategy(self, strategy_id: str, params: AdaptiveParameters,
                       performance: 'StrategyPerformance', original_file: Path) -> Optional[Path]:
        """
        Recode a strategy with improved parameters and save as new file.
        Returns the path to the new strategy file.
        """
        version = self.get_next_version(strategy_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create new filename with version - USE SAME NAMING AS SERVER!
        # Server uses: Strategy_v23.py (not Strategy_improved_v23.py)
        new_filename = f"{strategy_id}_v{version}.py"
        new_filepath = Config.IMPROVED_STRATEGIES_DIR / new_filename

        # Generate the improved strategy code
        strategy_code = self._generate_improved_strategy_code(
            strategy_id=strategy_id,
            params=params,
            performance=performance,
            original_file=original_file,
            version=version
        )

        # Save the new strategy file
        try:
            with open(new_filepath, 'w') as f:
                f.write(strategy_code)

            # Also save a metadata JSON file
            meta_filepath = Config.IMPROVED_STRATEGIES_DIR / f"{strategy_id}_v{version}_meta.json"
            meta_data = {
                'strategy_id': strategy_id,
                'version': version,
                'timestamp': timestamp,
                'original_file': str(original_file),
                'parameters': params.to_dict(),
                'performance_at_optimization': {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'profit_factor': performance.profit_factor,
                    'total_pnl': performance.total_pnl
                },
                'optimization_count': params.optimization_count
            }
            with open(meta_filepath, 'w') as f:
                json.dump(meta_data, f, indent=2)

            # Update improvement history
            if strategy_id not in self.improvement_history:
                self.improvement_history[strategy_id] = []
            self.improvement_history[strategy_id].append({
                'version': version,
                'timestamp': timestamp,
                'filepath': str(new_filepath),
                'parameters': params.to_dict()
            })
            self._save_history()

            print(f"📝 RECODED: {new_filename}")
            print(f"   Saved to: {new_filepath}")
            return new_filepath

        except Exception as e:
            print(f"❌ Failed to recode strategy: {e}")
            return None

    def _generate_improved_strategy_code(self, strategy_id: str, params: AdaptiveParameters,
                                         performance: 'StrategyPerformance',
                                         original_file: Path, version: int) -> str:
        """Generate Python code for the improved strategy"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        code = f'''#!/usr/bin/env python3
"""
IMPROVED STRATEGY - {strategy_id}
=================================================================
Version: {version}
Generated: {timestamp}
Original: {original_file.name}

This is an AUTO-IMPROVED version of the original strategy.
Parameters have been optimized based on live trading analysis.

OPTIMIZATION HISTORY:
- Optimization count: {params.optimization_count}
- Trades analyzed: {performance.total_trades}
- Win rate at optimization: {performance.win_rate:.1%}
- Profit factor at optimization: {performance.profit_factor:.2f}
=================================================================
"""

# =============================================================================
# OPTIMIZED PARAMETERS (learned from live trading)
# =============================================================================

class OptimizedParameters:
    """These parameters were learned from analyzing {performance.total_trades} live trades"""

    # Entry Conditions
    MIN_DEVIATION_PERCENT = {params.min_deviation_percent:.4f}   # Minimum deviation from band to enter
    MAX_VOLATILITY_PERCENT = {params.max_volatility_percent:.4f}  # Max 24h volatility to enter
    MIN_VOLUME_RATIO = {params.min_volume_ratio:.4f}             # Min volume vs average

    # Risk Management
    STOP_LOSS_ATR_MULTIPLIER = {params.stop_loss_atr_multiplier:.4f}
    TAKE_PROFIT_ATR_MULTIPLIER = {params.take_profit_atr_multiplier:.4f}
    MAX_HOLDING_PERIODS = {params.max_holding_periods}           # Max periods before forced exit

    # Band Calculation
    STD_DEV_MULTIPLIER = {params.std_dev_multiplier:.4f}         # For VWAP bands
    LOOKBACK_PERIOD = {params.lookback_period}                   # For rolling calculations

    # Position Sizing
    POSITION_SIZE_PERCENT = {params.position_size_percent:.4f}   # % of capital per trade


# =============================================================================
# STRATEGY IMPLEMENTATION
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional

class ImprovedStrategy:
    """
    Improved version of: {strategy_id}

    Changes from original:
    - min_deviation: {params.min_deviation_percent:.2f}%
    - max_volatility: {params.max_volatility_percent:.2f}%
    - stop_loss_mult: {params.stop_loss_atr_multiplier:.2f}
    - take_profit_mult: {params.take_profit_atr_multiplier:.2f}
    - std_dev_mult: {params.std_dev_multiplier:.2f}
    """

    def __init__(self):
        self.params = OptimizedParameters()

    def calculate_vwap_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate VWAP and bands with optimized parameters"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_vp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        vwap = cumulative_vp / cumulative_volume

        price_deviation = np.abs(df['Close'] - vwap)
        std_dev = price_deviation.rolling(window=self.params.LOOKBACK_PERIOD).std()

        upper_band = vwap + (self.params.STD_DEV_MULTIPLIER * std_dev)
        lower_band = vwap - (self.params.STD_DEV_MULTIPLIER * std_dev)

        return {{
            'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0.0,
            'upper_band': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
            'lower_band': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0,
            'std_dev': float(std_dev.iloc[-1]) if not pd.isna(std_dev.iloc[-1]) else 0.0
        }}

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate 24h volatility"""
        returns = df['Close'].pct_change()
        volatility_24h = returns.tail(96).std() * 100 * np.sqrt(96)
        return float(volatility_24h) if not pd.isna(volatility_24h) else 0.0

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Generate trading signal with optimized parameters"""
        bands = self.calculate_vwap_bands(df)
        atr = self.calculate_atr(df)
        volatility = self.calculate_volatility(df)

        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Calculate deviations
        lower_deviation = 0.0
        upper_deviation = 0.0

        if bands['lower_band'] > 0:
            lower_deviation = (bands['lower_band'] - current_price) / current_price * 100
        if bands['upper_band'] > 0:
            upper_deviation = (current_price - bands['upper_band']) / current_price * 100

        # Check filters
        volatility_ok = volatility < self.params.MAX_VOLATILITY_PERCENT

        # BUY signal
        if (current_price < bands['lower_band'] and
            lower_deviation > self.params.MIN_DEVIATION_PERCENT and
            volatility_ok):

            signal = "BUY"
            reason = f"Price below lower band (dev: {{lower_deviation:.2f}}%)"
            target_price = current_price + (atr * self.params.TAKE_PROFIT_ATR_MULTIPLIER)
            stop_loss = current_price - (atr * self.params.STOP_LOSS_ATR_MULTIPLIER)

        # SELL signal
        elif (current_price > bands['upper_band'] and
              upper_deviation > self.params.MIN_DEVIATION_PERCENT and
              volatility_ok):

            signal = "SELL"
            reason = f"Price above upper band (dev: {{upper_deviation:.2f}}%)"
            target_price = current_price - (atr * self.params.TAKE_PROFIT_ATR_MULTIPLIER)
            stop_loss = current_price + (atr * self.params.STOP_LOSS_ATR_MULTIPLIER)

        return {{
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'vwap': bands['vwap'],
            'upper_band': bands['upper_band'],
            'lower_band': bands['lower_band'],
            'atr': atr,
            'volatility': volatility,
            'deviation_percent': max(lower_deviation, upper_deviation)
        }}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Improved Strategy: {strategy_id}")
    print(f"Version: {version}")
    print(f"Optimized parameters loaded from live trading analysis")

    strategy = ImprovedStrategy()
    print(f"Min deviation: {{strategy.params.MIN_DEVIATION_PERCENT:.2f}}%")
    print(f"Max volatility: {{strategy.params.MAX_VOLATILITY_PERCENT:.2f}}%")
    print(f"Stop loss multiplier: {{strategy.params.STOP_LOSS_ATR_MULTIPLIER:.2f}}")
    print(f"Take profit multiplier: {{strategy.params.TAKE_PROFIT_ATR_MULTIPLIER:.2f}}")
'''
        return code

# =============================================================================
# LLM-BASED STRATEGY OPTIMIZER (LIKE APEX RBI)
# =============================================================================

class LLMStrategyOptimizer:
    """
    Uses LLM reasoning to analyze why strategies are losing and suggest improvements.
    This is the same approach as APEX RBI (Reasoning-Based Iteration).

    Instead of hardcoded rules like "multiply by 1.2", the LLM:
    1. Analyzes the trade history and loss patterns
    2. Reasons about WHY the strategy is failing
    3. Suggests specific parameter changes
    4. Can even recode the entire strategy if needed
    """

    def __init__(self):
        self.optimization_history: List[Dict] = []
        self._check_llm_availability()

    def _check_llm_availability(self):
        """Check which LLM providers are available"""
        self.available_providers = []

        if Config.DEEPSEEK_API_KEY:
            self.available_providers.append("deepseek")
            print("✅ DeepSeek API available for reasoning")
        if Config.XAI_API_KEY:
            self.available_providers.append("xai")
            print("✅ XAI (Grok) API available for reasoning")
        if Config.OPENAI_API_KEY and openai:
            self.available_providers.append("openai")
            print("✅ OpenAI API available for reasoning")
        if Config.ANTHROPIC_API_KEY and anthropic:
            self.available_providers.append("anthropic")
            print("✅ Anthropic API available for reasoning")

        if not self.available_providers:
            print("⚠️  No LLM API keys found - will use fallback heuristic optimization")
            print("   Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for LLM reasoning")

    def call_llm(self, prompt: str, system_prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Call LLM with fallback through available providers"""

        for provider in self.available_providers:
            try:
                if provider == "deepseek":
                    return self._call_deepseek(prompt, system_prompt, temperature)
                elif provider == "openai":
                    return self._call_openai(prompt, system_prompt, temperature)
                elif provider == "anthropic":
                    return self._call_anthropic(prompt, system_prompt, temperature)
                elif provider == "xai":
                    return self._call_xai(prompt, system_prompt, temperature)
            except Exception as e:
                print(f"⚠️  {provider} failed: {e}, trying next provider...")
                continue

        return None

    def _call_deepseek(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call DeepSeek API (same pattern as apex.py)"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _call_openai(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call OpenAI API"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call Anthropic API"""
        if not anthropic:
            raise ImportError("anthropic package not installed")

        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _call_xai(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call XAI (Grok) API"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.OpenAI(
            api_key=Config.XAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )

        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def analyze_and_optimize(self, strategy_id: str, params: 'AdaptiveParameters',
                            performance: 'StrategyPerformance',
                            recent_trades: List['TradeContext'],
                            original_strategy_code: str = "") -> Tuple[Optional['AdaptiveParameters'], Optional[str]]:
        """
        Use LLM to analyze strategy performance and suggest improvements.
        Returns: (optimized_params, optimized_code) or (None, None) if optimization failed
        """

        if not self.available_providers:
            print("⚠️  No LLM available, using fallback heuristics")
            return self._fallback_optimize(params, performance), None

        # Build the analysis prompt
        system_prompt = """You are an expert quantitative trading strategist and Python developer.
Your task is to analyze why a trading strategy is underperforming and suggest SPECIFIC improvements.

You must respond in a STRICT JSON format with the following structure:
{
    "analysis": "Your detailed analysis of why the strategy is failing",
    "root_causes": ["cause1", "cause2", ...],
    "parameter_changes": {
        "min_deviation_percent": <new_value or null if no change>,
        "max_volatility_percent": <new_value or null if no change>,
        "stop_loss_atr_multiplier": <new_value or null if no change>,
        "take_profit_atr_multiplier": <new_value or null if no change>,
        "std_dev_multiplier": <new_value or null if no change>,
        "min_volume_ratio": <new_value or null if no change>,
        "max_holding_periods": <new_value or null if no change>
    },
    "reasoning": "Explain WHY each parameter change will help",
    "confidence": <0.0 to 1.0>,
    "needs_full_recode": <true/false>
}

Be specific with numbers. Don't suggest vague changes like "increase slightly" - give exact values."""

        # Build trade history summary
        trade_summary = self._build_trade_summary(recent_trades)

        user_prompt = f"""Analyze this underperforming trading strategy and suggest improvements:

## STRATEGY: {strategy_id}

## CURRENT PARAMETERS:
- min_deviation_percent: {params.min_deviation_percent:.4f} (minimum price deviation from band to enter)
- max_volatility_percent: {params.max_volatility_percent:.4f} (maximum 24h volatility to trade)
- stop_loss_atr_multiplier: {params.stop_loss_atr_multiplier:.4f} (stop loss = ATR * this)
- take_profit_atr_multiplier: {params.take_profit_atr_multiplier:.4f} (target = ATR * this)
- std_dev_multiplier: {params.std_dev_multiplier:.4f} (for VWAP band calculation)
- min_volume_ratio: {params.min_volume_ratio:.4f} (minimum volume vs average)
- max_holding_periods: {params.max_holding_periods} (max 15-min periods before forced exit)

## PERFORMANCE METRICS:
- Total Trades: {performance.total_trades}
- Win Rate: {performance.win_rate:.1%} (TARGET: {Config.TARGET_WIN_RATE:.0%})
- Profit Factor: {performance.profit_factor:.2f} (TARGET: {Config.TARGET_PROFIT_FACTOR})
- Total PnL: ${performance.total_pnl:.2f}
- Consecutive Losses: {performance.consecutive_losses}
- Stop Loss Hits: {performance.stop_loss_hits}
- Target Hits: {performance.target_hits}
- Losses from low deviation entries: {performance.losses_low_deviation}
- Losses from high volatility: {performance.losses_high_volatility}

## RECENT TRADE HISTORY:
{trade_summary}

## YOUR TASK:
1. Analyze WHY this strategy is losing money
2. Identify the ROOT CAUSES (not just symptoms)
3. Suggest SPECIFIC parameter changes with EXACT numbers
4. Explain your reasoning

Respond ONLY with valid JSON."""

        try:
            print(f"\n🤖 Sending to LLM for analysis...")
            response = self.call_llm(user_prompt, system_prompt)

            if not response:
                print("❌ LLM returned empty response")
                return self._fallback_optimize(params, performance), None

            # Parse LLM response
            optimized_params, analysis = self._parse_llm_response(response, params)

            if optimized_params:
                print(f"✅ LLM Analysis Complete:")
                print(f"   {analysis.get('analysis', 'No analysis')[:200]}...")
                print(f"   Confidence: {analysis.get('confidence', 0):.0%}")

                # Log this optimization
                self.optimization_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'strategy_id': strategy_id,
                    'analysis': analysis,
                    'old_params': params.to_dict(),
                    'new_params': optimized_params.to_dict()
                })

                return optimized_params, None
            else:
                print("⚠️  Could not parse LLM response, using fallback")
                return self._fallback_optimize(params, performance), None

        except Exception as e:
            print(f"❌ LLM optimization failed: {e}")
            return self._fallback_optimize(params, performance), None

    def _build_trade_summary(self, trades: List['TradeContext']) -> str:
        """Build a detailed summary of recent trades for the LLM"""
        if not trades:
            return "No trades yet"

        summary_lines = []

        # Overall statistics
        total_trades = len(trades)
        wins = len([t for t in trades if t.pnl_usd > 0])
        losses = len([t for t in trades if t.pnl_usd <= 0])
        total_pnl = sum(t.pnl_usd for t in trades)
        avg_win = sum(t.pnl_usd for t in trades if t.pnl_usd > 0) / wins if wins > 0 else 0
        avg_loss = sum(t.pnl_usd for t in trades if t.pnl_usd <= 0) / losses if losses > 0 else 0

        summary_lines.append(f"### TRADE STATISTICS ({total_trades} trades)")
        summary_lines.append(f"- Wins: {wins}, Losses: {losses}")
        summary_lines.append(f"- Total PnL: ${total_pnl:+.2f}")
        summary_lines.append(f"- Average Win: ${avg_win:+.2f}, Average Loss: ${avg_loss:.2f}")
        summary_lines.append("")

        # Market conditions analysis
        market_analysis = self._analyze_market_conditions(trades)
        summary_lines.append("### MARKET CONDITIONS ANALYSIS")
        for key, value in market_analysis.items():
            summary_lines.append(f"- {key}: {value}")
        summary_lines.append("")

        # Individual trade details (last 10)
        summary_lines.append("### RECENT TRADES (last 10)")
        for i, trade in enumerate(trades[-10:], 1):
            outcome = "WIN ✅" if trade.pnl_usd > 0 else "LOSS ❌"
            holding_time = getattr(trade, 'holding_periods', 'N/A')
            summary_lines.append(
                f"{i}. {trade.symbol} {trade.direction}: "
                f"Entry ${trade.entry_price:.2f} → Exit ${trade.exit_price:.2f}, "
                f"PnL: ${trade.pnl_usd:+.2f} ({outcome})"
            )
            summary_lines.append(
                f"   Exit Reason: {trade.exit_reason}, "
                f"Deviation: {trade.deviation_percent:.2f}%, "
                f"Volatility: {trade.volatility_24h:.2f}%, "
                f"Trend: {trade.trend_direction}, "
                f"Holding: {holding_time} periods"
            )

        return "\n".join(summary_lines)

    def _analyze_market_conditions(self, trades: List['TradeContext']) -> Dict[str, str]:
        """Analyze market conditions from recent trades for better LLM context"""
        if not trades:
            return {}

        analysis = {}

        # Trend analysis
        up_trends = sum(1 for t in trades if t.trend_direction == "UP")
        down_trends = sum(1 for t in trades if t.trend_direction == "DOWN")
        sideways = sum(1 for t in trades if t.trend_direction == "SIDEWAYS")
        total = len(trades)
        analysis["Dominant Trend"] = f"UP: {up_trends}/{total}, DOWN: {down_trends}/{total}, SIDEWAYS: {sideways}/{total}"

        # Pre-compute losing trades to avoid duplicate list comprehension
        losing_trades = [t for t in trades if t.pnl_usd <= 0]
        num_losses = len(losing_trades)

        # Volatility analysis
        avg_volatility = sum(t.volatility_24h for t in trades) / total
        high_vol_losses = sum(1 for t in losing_trades if t.volatility_24h > avg_volatility)
        analysis["Avg Volatility"] = f"{avg_volatility:.2f}%"
        analysis["High Vol Losses"] = f"{high_vol_losses}/{num_losses} losses in high volatility"

        # Deviation analysis
        avg_deviation = sum(t.deviation_percent for t in trades) / total
        low_dev_losses = sum(1 for t in losing_trades if t.deviation_percent < avg_deviation)
        analysis["Avg Entry Deviation"] = f"{avg_deviation:.2f}%"
        analysis["Low Dev Losses"] = f"{low_dev_losses}/{num_losses} losses from low deviation entries"

        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'wins': 0, 'losses': 0}
            if trade.pnl_usd > 0:
                exit_reasons[reason]['wins'] += 1
            else:
                exit_reasons[reason]['losses'] += 1

        exit_summary = []
        for reason, counts in exit_reasons.items():
            exit_summary.append(f"{reason}: {counts['wins']}W/{counts['losses']}L")
        analysis["Exit Reasons"] = ", ".join(exit_summary)

        # Win rate by direction
        buy_trades = [t for t in trades if t.direction == "BUY"]
        sell_trades = [t for t in trades if t.direction == "SELL"]
        buy_wins = sum(1 for t in buy_trades if t.pnl_usd > 0) if buy_trades else 0
        sell_wins = sum(1 for t in sell_trades if t.pnl_usd > 0) if sell_trades else 0
        analysis["Direction Performance"] = f"BUY: {buy_wins}/{len(buy_trades)} wins, SELL: {sell_wins}/{len(sell_trades)} wins"

        # NEW: Regime performance analysis
        regime_perf = self._build_regime_trade_summary(trades)
        analysis["Regime Performance"] = regime_perf

        return analysis

    def _build_regime_trade_summary(self, trades: List['TradeContext']) -> str:
        """Build summary of trades by market regime for LLM context"""
        if not trades:
            return "No trades yet"

        regime_performance = {}
        for trade in trades[-20:]:  # Last 20 trades
            regime = getattr(trade, 'market_regime', 'UNKNOWN')
            if not regime:
                regime = 'UNKNOWN'
            if regime not in regime_performance:
                regime_performance[regime] = {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0}

            regime_performance[regime]['total'] += 1
            regime_performance[regime]['pnl'] += trade.pnl_usd
            if trade.pnl_usd > 0:
                regime_performance[regime]['wins'] += 1
            else:
                regime_performance[regime]['losses'] += 1

        summary_lines = []
        for regime, stats in regime_performance.items():
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            summary_lines.append(
                f"  - {regime}: {stats['wins']}W/{stats['losses']}L "
                f"({win_rate:.1%} WR, ${stats['pnl']:+.2f} PnL)"
            )

        return "\n".join(summary_lines) if summary_lines else "No regime data available"

    def _parse_llm_response(self, response: str, current_params: 'AdaptiveParameters') -> Tuple[Optional['AdaptiveParameters'], Dict]:
        """Parse LLM JSON response and create new parameters"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None, {}

            analysis = json.loads(json_match.group())

            # Create new params based on LLM suggestions
            new_params = AdaptiveParameters(
                min_deviation_percent=current_params.min_deviation_percent,
                max_volatility_percent=current_params.max_volatility_percent,
                stop_loss_atr_multiplier=current_params.stop_loss_atr_multiplier,
                take_profit_atr_multiplier=current_params.take_profit_atr_multiplier,
                std_dev_multiplier=current_params.std_dev_multiplier,
                min_volume_ratio=current_params.min_volume_ratio,
                max_holding_periods=current_params.max_holding_periods,
                lookback_period=current_params.lookback_period,
                position_size_percent=current_params.position_size_percent,
                version=current_params.version + 1,
                optimization_count=current_params.optimization_count + 1
            )

            # Apply LLM suggested changes
            param_changes = analysis.get('parameter_changes', {})

            if param_changes.get('min_deviation_percent') is not None:
                new_params.min_deviation_percent = float(param_changes['min_deviation_percent'])
                print(f"   📈 min_deviation: {current_params.min_deviation_percent:.2f} → {new_params.min_deviation_percent:.2f}")

            if param_changes.get('max_volatility_percent') is not None:
                new_params.max_volatility_percent = float(param_changes['max_volatility_percent'])
                print(f"   📉 max_volatility: {current_params.max_volatility_percent:.2f} → {new_params.max_volatility_percent:.2f}")

            if param_changes.get('stop_loss_atr_multiplier') is not None:
                new_params.stop_loss_atr_multiplier = float(param_changes['stop_loss_atr_multiplier'])
                print(f"   🛑 stop_loss_mult: {current_params.stop_loss_atr_multiplier:.2f} → {new_params.stop_loss_atr_multiplier:.2f}")

            if param_changes.get('take_profit_atr_multiplier') is not None:
                new_params.take_profit_atr_multiplier = float(param_changes['take_profit_atr_multiplier'])
                print(f"   🎯 take_profit_mult: {current_params.take_profit_atr_multiplier:.2f} → {new_params.take_profit_atr_multiplier:.2f}")

            if param_changes.get('std_dev_multiplier') is not None:
                new_params.std_dev_multiplier = float(param_changes['std_dev_multiplier'])
                print(f"   📊 std_dev_mult: {current_params.std_dev_multiplier:.2f} → {new_params.std_dev_multiplier:.2f}")

            if param_changes.get('min_volume_ratio') is not None:
                new_params.min_volume_ratio = float(param_changes['min_volume_ratio'])
                print(f"   📊 min_volume: {current_params.min_volume_ratio:.2f} → {new_params.min_volume_ratio:.2f}")

            if param_changes.get('max_holding_periods') is not None:
                new_params.max_holding_periods = int(param_changes['max_holding_periods'])
                print(f"   ⏱️  max_holding: {current_params.max_holding_periods} → {new_params.max_holding_periods}")

            return new_params, analysis

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM JSON: {e}")
            return None, {}
        except Exception as e:
            print(f"❌ Error parsing LLM response: {e}")
            return None, {}

    def _fallback_optimize(self, params: 'AdaptiveParameters',
                          performance: 'StrategyPerformance') -> 'AdaptiveParameters':
        """Fallback heuristic optimization when no LLM is available"""
        print("📊 Using fallback heuristic optimization...")

        new_params = AdaptiveParameters(
            min_deviation_percent=params.min_deviation_percent,
            max_volatility_percent=params.max_volatility_percent,
            stop_loss_atr_multiplier=params.stop_loss_atr_multiplier,
            take_profit_atr_multiplier=params.take_profit_atr_multiplier,
            std_dev_multiplier=params.std_dev_multiplier,
            min_volume_ratio=params.min_volume_ratio,
            max_holding_periods=params.max_holding_periods,
            lookback_period=params.lookback_period,
            position_size_percent=params.position_size_percent,
            version=params.version + 1,
            optimization_count=params.optimization_count + 1
        )

        # Simple heuristics based on loss patterns
        if performance.losses_low_deviation > performance.total_trades * 0.3:
            new_params.min_deviation_percent *= 1.3
            print(f"   📈 Increased min_deviation (many low deviation losses)")

        if performance.losses_high_volatility > performance.total_trades * 0.3:
            new_params.max_volatility_percent *= 0.7
            print(f"   📉 Reduced max_volatility (many high volatility losses)")

        if performance.stop_loss_hits > performance.target_hits * 2:
            new_params.stop_loss_atr_multiplier *= 1.3
            print(f"   🛑 Widened stops (too many stop hits)")

        if performance.target_hits < performance.total_trades * 0.2:
            new_params.take_profit_atr_multiplier *= 0.8
            print(f"   🎯 Reduced targets (too few target hits)")

        return new_params

# =============================================================================
# HTX API CLIENT
# =============================================================================

class RealHTXClient:
    def __init__(self):
        self.base_url = Config.HTX_BASE_URL
        self.session = requests.Session()

    def fetch_candles(self, symbol: str, period: str = '15min', count: int = 100) -> Optional[pd.DataFrame]:
        try:
            endpoint = f"{self.base_url}/market/history/kline"
            params = {
                "symbol": f"{symbol.lower()}usdt",
                "period": period,
                "size": min(count, 2000)
            }

            response = self.session.get(endpoint, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    klines = data.get("data", [])

                    if klines:
                        df = pd.DataFrame(klines)
                        df = df.rename(columns={
                            'id': 'timestamp',
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'amount': 'Volume'
                        })
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        df = df.sort_values('datetime')
                        return df

        except Exception as e:
            print(f"❌ HTX API Error for {symbol}: {e}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            candles = self.fetch_candles(symbol, '1min', 1)
            if candles is not None and len(candles) > 0:
                return float(candles['Close'].iloc[-1])
        except:
            pass
        return None

# =============================================================================
# LLM-BASED SIGNAL GENERATOR - READS STRATEGY CODE TO UNDERSTAND WHAT TO TRADE
# =============================================================================

class LLMSignalGenerator:
    """
    Uses LLM to read strategy code and generate appropriate trading signals.

    Instead of hardcoding VWAP logic for all strategies, this class:
    1. Reads the actual strategy Python file
    2. Asks LLM to understand what signals the strategy looks for
    3. Generates signals based on the strategy's actual logic
    """

    def __init__(self, llm_optimizer: 'LLMStrategyOptimizer'):
        self.llm_optimizer = llm_optimizer
        self.strategy_cache: Dict[str, Dict] = {}  # Cache parsed strategy info

    def parse_strategy_logic(self, strategy_file: Path) -> Dict:
        """Read strategy file and use LLM to understand its trading logic"""
        strategy_id = strategy_file.stem

        # Return cached result if available
        if strategy_id in self.strategy_cache:
            return self.strategy_cache[strategy_id]

        try:
            # Read strategy code
            with open(strategy_file, 'r') as f:
                code = f.read()

            # If no LLM available, extract basic info from code
            if not self.llm_optimizer.available_providers:
                return self._extract_basic_logic(code, strategy_id)

            # Ask LLM to understand the strategy
            system_prompt = """You are a quantitative trading expert analyzing Python trading strategies.
Your task is to understand what trading signals a strategy generates and extract the key conditions.

Return a JSON object with:
{
    "strategy_type": "MEAN_REVERSION" or "MOMENTUM" or "MARKET_MAKING" or "PAIRS_TRADING" or "ML_PREDICTION" or "OTHER",
    "entry_conditions": {
        "long": "description of when to buy",
        "short": "description of when to sell"
    },
    "key_indicators": ["list of indicators used like VWAP, RSI, MACD, etc"],
    "risk_management": {
        "stop_loss_logic": "how stop loss is calculated",
        "take_profit_logic": "how take profit is calculated"
    }
}"""

            user_prompt = f"""Analyze this trading strategy code and extract its trading logic:

```python
{code[:4000]}  # Truncate to avoid token limits
```

What conditions trigger BUY and SELL signals? Return JSON only."""

            response = self.llm_optimizer.call_llm(user_prompt, system_prompt, temperature=0.2)

            if response:
                try:
                    # Try to parse JSON from response
                    import json
                    # Find JSON in response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        parsed = json.loads(response[json_start:json_end])
                        parsed['code_snippet'] = code[:2000]
                        self.strategy_cache[strategy_id] = parsed
                        return parsed
                except:
                    pass

            # Fallback to basic extraction
            return self._extract_basic_logic(code, strategy_id)

        except Exception as e:
            print(f"⚠️  Could not parse strategy {strategy_id}: {e}")
            return {'strategy_type': 'UNKNOWN', 'key_indicators': []}

    def _extract_basic_logic(self, code: str, strategy_id: str) -> Dict:
        """Extract basic strategy info from code without LLM"""
        code_upper = code.upper()

        # Detect strategy type from code patterns
        if 'VWAP' in code_upper or 'MEAN_REVERSION' in code_upper:
            strategy_type = 'MEAN_REVERSION'
        elif 'INVENTORY' in code_upper or 'MARKET_MAKER' in code_upper or 'STOIKOV' in code_upper:
            strategy_type = 'MARKET_MAKING'
        elif 'NEURAL' in code_upper or 'MLP' in code_upper or 'LSTM' in code_upper or 'PREDICT' in code_upper:
            strategy_type = 'ML_PREDICTION'
        elif 'COINTEGRATION' in code_upper or 'PAIRS' in code_upper or 'Z_SCORE' in code_upper:
            strategy_type = 'PAIRS_TRADING'
        elif 'RSI' in code_upper or 'MACD' in code_upper or 'MOMENTUM' in code_upper:
            strategy_type = 'MOMENTUM'
        elif 'GAP' in code_upper or 'EARNINGS' in code_upper or 'REVERSAL' in code_upper:
            strategy_type = 'EVENT_DRIVEN'
        else:
            strategy_type = 'UNKNOWN'

        # Extract indicators mentioned
        indicators = []
        indicator_patterns = ['VWAP', 'RSI', 'MACD', 'ATR', 'SMA', 'EMA', 'BOLLINGER', 'STOCHASTIC', 'ADX', 'VOLUME']
        for ind in indicator_patterns:
            if ind in code_upper:
                indicators.append(ind)

        return {
            'strategy_type': strategy_type,
            'key_indicators': indicators,
            'code_snippet': code[:2000]
        }

    def generate_signal_for_strategy(self, strategy_file: Path, df: pd.DataFrame,
                                     current_price: float, params: 'AdaptiveParameters') -> Dict:
        """Generate signal using LLM understanding of strategy logic"""

        strategy_info = self.parse_strategy_logic(strategy_file)
        strategy_type = strategy_info.get('strategy_type', 'UNKNOWN')

        # Calculate common market metrics
        context = self._calculate_market_context(df, params)

        # Generate signal based on strategy type
        if strategy_type == 'MEAN_REVERSION':
            return self._generate_mean_reversion_signal(df, current_price, context, params)
        elif strategy_type == 'MARKET_MAKING':
            return self._generate_market_making_signal(df, current_price, context, params)
        elif strategy_type == 'ML_PREDICTION':
            return self._generate_ml_prediction_signal(df, current_price, context, params)
        elif strategy_type == 'PAIRS_TRADING':
            return self._generate_pairs_trading_signal(df, current_price, context, params)
        elif strategy_type == 'MOMENTUM':
            return self._generate_momentum_signal(df, current_price, context, params)
        elif strategy_type == 'EVENT_DRIVEN':
            return self._generate_event_driven_signal(df, current_price, context, params)
        else:
            # Default to mean reversion for unknown types
            return self._generate_mean_reversion_signal(df, current_price, context, params)

    def _calculate_market_context(self, df: pd.DataFrame, params: 'AdaptiveParameters') -> Dict:
        """Calculate comprehensive market context"""
        # VWAP and Bands
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_vp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        vwap = cumulative_vp / cumulative_volume

        price_deviation = np.abs(df['Close'] - vwap)
        std_dev = price_deviation.rolling(window=params.lookback_period).std()

        upper_band = vwap + (params.std_dev_multiplier * std_dev)
        lower_band = vwap - (params.std_dev_multiplier * std_dev)

        # ATR
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()

        # Volatility
        returns = df['Close'].pct_change()
        volatility_24h = returns.tail(96).std() * 100 * np.sqrt(96)

        # Volume Ratio
        avg_volume = df['Volume'].rolling(window=20).mean()
        volume_ratio = df['Volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0

        # Trend
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_5 = df['Close'].rolling(window=5).mean()
        if ma_5.iloc[-1] > ma_20.iloc[-1] * 1.005:
            trend = "UP"
        elif ma_5.iloc[-1] < ma_20.iloc[-1] * 0.995:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        return {
            'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0.0,
            'upper_band': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
            'lower_band': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0,
            'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            'volatility_24h': float(volatility_24h) if not pd.isna(volatility_24h) else 0.0,
            'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
            'trend': trend
        }

    def _generate_mean_reversion_signal(self, df: pd.DataFrame, current_price: float,
                                        context: Dict, params: 'AdaptiveParameters') -> Dict:
        """Mean reversion: Buy oversold, sell overbought
        
        🔥 TREND-BASED TRADING:
        - Trend DOWN → SELL/SHORT (profit from falling prices!)
        - Trend UP → BUY/LONG (profit from rising prices!)
        - Don't fight the trend - trade WITH it!
        """
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        lower_dev = (context['lower_band'] - current_price) / current_price * 100 if context['lower_band'] > 0 else 0
        upper_dev = (current_price - context['upper_band']) / current_price * 100 if context['upper_band'] > 0 else 0
        
        # Get trend from context
        trend = context.get('trend', 'SIDEWAYS')
        
        # 🔥 TREND FILTER: Trade WITH the trend, not against it!
        enable_trend_filter = getattr(Config, 'ENABLE_TREND_FILTER', True)

        # Original mean reversion logic (price below lower band = oversold)
        if current_price < context['lower_band'] and lower_dev > params.min_deviation_percent:
            if enable_trend_filter and trend == "DOWN":
                # 🔥 Trend is DOWN - don't BUY into falling knife, SELL instead!
                signal = "SELL"
                reason = f"TREND SELL: Price ${current_price:.2f} falling, trend DOWN - SHORT for profit!"
                target_price = current_price - (context['atr'] * params.take_profit_atr_multiplier)
                stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)
            else:
                # Trend is UP or SIDEWAYS - safe to BUY the dip
                signal = "BUY"
                reason = f"Mean Reversion BUY: Price ${current_price:.2f} below band (trend: {trend})"
                target_price = current_price + (context['atr'] * params.take_profit_atr_multiplier)
                stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
                
        # Original mean reversion logic (price above upper band = overbought)
        elif current_price > context['upper_band'] and upper_dev > params.min_deviation_percent:
            if enable_trend_filter and trend == "UP":
                # 🔥 Trend is UP - don't SELL into rally, BUY instead!
                signal = "BUY"
                reason = f"TREND BUY: Price ${current_price:.2f} rising, trend UP - LONG for profit!"
                target_price = current_price + (context['atr'] * params.take_profit_atr_multiplier)
                stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
            else:
                # Trend is DOWN or SIDEWAYS - safe to SELL the top
                signal = "SELL"
                reason = f"Mean Reversion SELL: Price ${current_price:.2f} above band (trend: {trend})"
                target_price = current_price - (context['atr'] * params.take_profit_atr_multiplier)
                stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': max(lower_dev, upper_dev), **context}

    def _generate_market_making_signal(self, df: pd.DataFrame, current_price: float,
                                       context: Dict, params: 'AdaptiveParameters') -> Dict:
        """Market making: Trade around mid-price with inventory management
        
        🔥 TREND-BASED TRADING:
        - Trend DOWN → SELL/SHORT
        - Trend UP → BUY/LONG
        """
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Spread-based entry
        spread = (df['High'].iloc[-1] - df['Low'].iloc[-1]) / current_price * 100
        mid_price = (df['High'].iloc[-1] + df['Low'].iloc[-1]) / 2
        price_vs_mid = (current_price - mid_price) / mid_price * 100
        
        # Get trend from context
        trend = context.get('trend', 'SIDEWAYS')
        enable_trend_filter = getattr(Config, 'ENABLE_TREND_FILTER', True)

        # Price below mid with good spread
        if price_vs_mid < -0.1 and spread > 0.2 and context['volume_ratio'] > 0.5:
            if enable_trend_filter and trend == "DOWN":
                # 🔥 Trend DOWN - SELL instead of BUY!
                signal = "SELL"
                reason = f"TREND SELL: Price ${current_price:.2f} falling, trend DOWN - SHORT!"
                target_price = current_price - (context['atr'] * 0.5)
                stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)
            else:
                signal = "BUY"
                reason = f"Market Making BUY: Price ${current_price:.2f} below mid (trend: {trend})"
                target_price = mid_price + (context['atr'] * 0.5)
                stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
                
        # Price above mid with good spread
        elif price_vs_mid > 0.1 and spread > 0.2 and context['volume_ratio'] > 0.5:
            if enable_trend_filter and trend == "UP":
                # 🔥 Trend UP - BUY instead of SELL!
                signal = "BUY"
                reason = f"TREND BUY: Price ${current_price:.2f} rising, trend UP - LONG!"
                target_price = current_price + (context['atr'] * 0.5)
                stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
            else:
                signal = "SELL"
                reason = f"Market Making SELL: Price ${current_price:.2f} above mid (trend: {trend})"
                target_price = mid_price - (context['atr'] * 0.5)
                stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': abs(price_vs_mid), **context}

    def _generate_ml_prediction_signal(self, df: pd.DataFrame, current_price: float,
                                       context: Dict, params: 'AdaptiveParameters') -> Dict:
        """ML-based: Use multiple indicators for prediction"""
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Multi-indicator scoring
        score = 0

        # RSI signals
        if context['rsi'] < 30:
            score += 2  # Oversold
        elif context['rsi'] > 70:
            score -= 2  # Overbought

        # MACD signals
        if context['macd'] > context['macd_signal']:
            score += 1  # Bullish crossover
        else:
            score -= 1  # Bearish crossover

        # Trend alignment
        if context['trend'] == 'UP':
            score += 1
        elif context['trend'] == 'DOWN':
            score -= 1

        # Volume confirmation
        if context['volume_ratio'] > 1.2:
            score = int(score * 1.2)  # Amplify signal on high volume

        if score >= 3:
            signal = "BUY"
            reason = f"ML Prediction: Bullish score {score} (RSI: {context['rsi']:.1f}, MACD: bullish)"
            target_price = current_price + (context['atr'] * params.take_profit_atr_multiplier)
            stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
        elif score <= -3:
            signal = "SELL"
            reason = f"ML Prediction: Bearish score {score} (RSI: {context['rsi']:.1f}, MACD: bearish)"
            target_price = current_price - (context['atr'] * params.take_profit_atr_multiplier)
            stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': abs(score), **context}

    def _generate_pairs_trading_signal(self, df: pd.DataFrame, current_price: float,
                                       context: Dict, params: 'AdaptiveParameters') -> Dict:
        """Pairs trading: Z-score based mean reversion"""
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Calculate z-score of price vs moving average
        ma_50 = df['Close'].rolling(window=50).mean()
        std_50 = df['Close'].rolling(window=50).std()
        z_score = (current_price - ma_50.iloc[-1]) / std_50.iloc[-1] if std_50.iloc[-1] > 0 else 0

        # Entry on extreme z-scores
        if z_score < -2.0:
            signal = "BUY"
            reason = f"Pairs/Z-Score: Z={z_score:.2f} (oversold, expecting reversion)"
            target_price = ma_50.iloc[-1]  # Target mean
            stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier * 1.5)
        elif z_score > 2.0:
            signal = "SELL"
            reason = f"Pairs/Z-Score: Z={z_score:.2f} (overbought, expecting reversion)"
            target_price = ma_50.iloc[-1]  # Target mean
            stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier * 1.5)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': abs(z_score), **context}

    def _generate_momentum_signal(self, df: pd.DataFrame, current_price: float,
                                  context: Dict, params: 'AdaptiveParameters') -> Dict:
        """Momentum: Follow the trend"""
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Calculate momentum
        momentum_5 = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
        momentum_20 = (current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100

        # Strong momentum with confirmation
        if momentum_5 > 1.0 and momentum_20 > 2.0 and context['rsi'] < 70 and context['volume_ratio'] > 1.0:
            signal = "BUY"
            reason = f"Momentum: Strong uptrend (5d: {momentum_5:.2f}%, 20d: {momentum_20:.2f}%)"
            target_price = current_price + (context['atr'] * params.take_profit_atr_multiplier * 1.5)
            stop_loss = current_price - (context['atr'] * params.stop_loss_atr_multiplier)
        elif momentum_5 < -1.0 and momentum_20 < -2.0 and context['rsi'] > 30 and context['volume_ratio'] > 1.0:
            signal = "SELL"
            reason = f"Momentum: Strong downtrend (5d: {momentum_5:.2f}%, 20d: {momentum_20:.2f}%)"
            target_price = current_price - (context['atr'] * params.take_profit_atr_multiplier * 1.5)
            stop_loss = current_price + (context['atr'] * params.stop_loss_atr_multiplier)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': abs(momentum_5), **context}

    def _generate_event_driven_signal(self, df: pd.DataFrame, current_price: float,
                                      context: Dict, params: 'AdaptiveParameters') -> Dict:
        """Event-driven: Gap reversals and unusual moves"""
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Calculate gap from previous close
        prev_close = df['Close'].iloc[-2]
        gap_percent = (df['Open'].iloc[-1] - prev_close) / prev_close * 100

        # Intraday reversal
        intraday_move = (current_price - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100

        # Gap down with reversal = buy
        if gap_percent < -1.0 and intraday_move > 0.3 and context['volume_ratio'] > 1.5:
            signal = "BUY"
            reason = f"Event: Gap down {gap_percent:.2f}% with reversal {intraday_move:.2f}%"
            target_price = prev_close  # Target gap fill
            stop_loss = df['Low'].iloc[-1] - (context['atr'] * 0.5)
        # Gap up with reversal = sell
        elif gap_percent > 1.0 and intraday_move < -0.3 and context['volume_ratio'] > 1.5:
            signal = "SELL"
            reason = f"Event: Gap up {gap_percent:.2f}% with reversal {intraday_move:.2f}%"
            target_price = prev_close  # Target gap fill
            stop_loss = df['High'].iloc[-1] + (context['atr'] * 0.5)

        return {'signal': signal, 'reason': reason, 'target_price': target_price, 'stop_loss': stop_loss,
                'current_price': current_price, 'deviation_percent': abs(gap_percent), **context}


# =============================================================================
# ADAPTIVE STRATEGY EXECUTOR
# =============================================================================

class AdaptiveStrategyExecutor:
    """Strategy executor with adaptive parameters and market regime awareness"""

    def __init__(self, params: AdaptiveParameters):
        self.params = params
        self.current_regime = "MIXED"  # Track current market regime
        self.strategy_type = "UNKNOWN"  # Will be set when strategy file is analyzed
        self.strategy_file: Optional[Path] = None

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime for adaptive trading.

        Regimes:
        - CHOPPY_HIGH_VOL: High volatility with no clear trend (avoid mean reversion)
        - STRONG_TREND: Low volatility with clear trend (use momentum/trend following)
        - RANGING_LOW_VOL: Low volatility, no trend (good for mean reversion)
        - MIXED: Standard conditions

        Returns:
            str: Market regime classification
        """
        try:
            # Check if we have enough data
            if len(df) < 20:
                return "MIXED"  # Not enough data for regime detection

            # Calculate returns and volatility (24h = 96 periods of 15 min each)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * 100 * np.sqrt(Config.PERIODS_PER_24H)

            # Calculate trend strength (ADX-like calculation)
            high, low, close = df['High'], df['Low'], df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = true_range.rolling(14).mean()

            # Moving averages for trend detection
            ma_20 = df['Close'].rolling(20).mean()
            ma_50 = df['Close'].rolling(50).mean() if len(df) >= 50 else ma_20

            # Safe trend strength calculation - use current price as fallback
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 1.0
            current_price = df['Close'].iloc[-1]
            ma20_val = ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else current_price
            ma50_val = ma_50.iloc[-1] if not pd.isna(ma_50.iloc[-1]) else ma20_val

            if current_atr > 0:
                trend_strength = abs(ma20_val - ma50_val) / current_atr
            else:
                trend_strength = 0.5  # Default to medium trend strength

            # Regime classification using Config thresholds
            if volatility > Config.REGIME_HIGH_VOL_THRESHOLD and trend_strength < Config.REGIME_WEAK_TREND_THRESHOLD:
                regime = "CHOPPY_HIGH_VOL"
            elif volatility < Config.REGIME_LOW_VOL_THRESHOLD and trend_strength > Config.REGIME_STRONG_TREND_THRESHOLD:
                regime = "STRONG_TREND"
            elif volatility < Config.REGIME_VERY_LOW_VOL_THRESHOLD and trend_strength < Config.REGIME_RANGING_TREND_THRESHOLD:
                regime = "RANGING_LOW_VOL"
            else:
                regime = "MIXED"

            self.current_regime = regime
            return regime

        except Exception as e:
            print(f"⚠️  Regime detection error: {e}")
            return "MIXED"

    def get_regime_adjusted_params(self, regime: str) -> Dict:
        """
        Get parameter adjustments based on current market regime.

        IMPORTANT: These are SMALL adjustments - NOT meant to prevent trading!
        The regime info is primarily for logging and LLM optimization context.
        Multipliers close to 1.0 to keep trading active.
        """
        adjustments = {
            'min_deviation_mult': 1.0,
            'stop_loss_mult': 1.0,
            'take_profit_mult': 1.0,
            'volatility_filter_mult': 1.0,
            'volume_filter_mult': 1.0,
            'regime_note': ''
        }

        if regime == "CHOPPY_HIGH_VOL":
            # High volatility - slightly wider stops, keep trading!
            adjustments['min_deviation_mult'] = 1.0   # NO CHANGE - don't block trades!
            adjustments['stop_loss_mult'] = 1.2       # Slightly wider stops
            adjustments['take_profit_mult'] = 1.1     # Slightly further targets
            adjustments['volatility_filter_mult'] = 1.5  # Allow higher volatility
            adjustments['regime_note'] = "⚡ CHOPPY: Wider stops"

        elif regime == "STRONG_TREND":
            # Strong trending - use normal params
            adjustments['min_deviation_mult'] = 1.0   # NO CHANGE
            adjustments['stop_loss_mult'] = 1.0       # Normal stops
            adjustments['take_profit_mult'] = 1.1     # Slightly further targets
            adjustments['regime_note'] = "📈 TRENDING"

        elif regime == "RANGING_LOW_VOL":
            # Low volatility ranging - perfect for mean reversion
            adjustments['min_deviation_mult'] = 0.9   # Slightly easier entry
            adjustments['stop_loss_mult'] = 1.0       # Normal stops
            adjustments['take_profit_mult'] = 0.9     # Closer targets (quick scalps)
            adjustments['regime_note'] = "📊 RANGING: Quick scalps"

        else:  # MIXED
            adjustments['regime_note'] = "⚖️ MIXED"

        return adjustments

    def calculate_market_context(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive market context"""
        # VWAP and Bands
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_vp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        vwap = cumulative_vp / cumulative_volume

        price_deviation = np.abs(df['Close'] - vwap)
        std_dev = price_deviation.rolling(window=self.params.lookback_period).std()

        upper_band = vwap + (self.params.std_dev_multiplier * std_dev)
        lower_band = vwap - (self.params.std_dev_multiplier * std_dev)

        # ATR
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=14).mean()

        # Volatility (24h = 96 15-min candles)
        returns = df['Close'].pct_change()
        volatility_24h = returns.tail(96).std() * 100 * np.sqrt(96)  # Annualized to daily

        # Volume Ratio (current vs 20-period average)
        avg_volume = df['Volume'].rolling(window=20).mean()
        volume_ratio = df['Volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0

        # Trend Direction (simple: compare current close to 20-period MA)
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_5 = df['Close'].rolling(window=5).mean()

        if ma_5.iloc[-1] > ma_20.iloc[-1] * 1.005:
            trend = "UP"
        elif ma_5.iloc[-1] < ma_20.iloc[-1] * 0.995:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        return {
            'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0.0,
            'upper_band': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
            'lower_band': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0,
            'std_dev': float(std_dev.iloc[-1]) if not pd.isna(std_dev.iloc[-1]) else 0.0,
            'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            'volatility_24h': float(volatility_24h) if not pd.isna(volatility_24h) else 0.0,
            'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
            'trend': trend
        }

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Generate trading signal with market regime awareness and adaptive parameters"""
        context = self.calculate_market_context(df)

        # NEW: Detect market regime and get adjustments
        regime = self.detect_market_regime(df)
        regime_adj = self.get_regime_adjusted_params(regime)

        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0

        # Apply regime adjustments to parameters
        effective_min_deviation = self.params.min_deviation_percent * regime_adj['min_deviation_mult']
        effective_max_volatility = self.params.max_volatility_percent * regime_adj['volatility_filter_mult']
        effective_stop_mult = self.params.stop_loss_atr_multiplier * regime_adj['stop_loss_mult']
        effective_target_mult = self.params.take_profit_atr_multiplier * regime_adj['take_profit_mult']
        effective_min_volume = self.params.min_volume_ratio * regime_adj['volume_filter_mult']

        # Calculate deviations
        lower_deviation = 0.0
        upper_deviation = 0.0

        if context['lower_band'] > 0:
            lower_deviation = (context['lower_band'] - current_price) / current_price * 100
        if context['upper_band'] > 0:
            upper_deviation = (current_price - context['upper_band']) / current_price * 100

        # Filter Conditions with regime-adjusted parameters
        volatility_ok = context['volatility_24h'] < effective_max_volatility
        volume_ok = context['volume_ratio'] >= effective_min_volume

        # BUY Signal - Price below lower band (with regime-adjusted thresholds)
        if (current_price < context['lower_band'] and
            lower_deviation > effective_min_deviation and
            volatility_ok and volume_ok):

            signal = "BUY"
            reason = f"Price ${current_price:.2f} below lower band ${context['lower_band']:.2f} (dev: {lower_deviation:.2f}%)"
            target_price = current_price + (context['atr'] * effective_target_mult)
            stop_loss = current_price - (context['atr'] * effective_stop_mult)

        # SELL Signal - Price above upper band (with regime-adjusted thresholds)
        elif (current_price > context['upper_band'] and
              upper_deviation > effective_min_deviation and
              volatility_ok and volume_ok):

            signal = "SELL"
            reason = f"Price ${current_price:.2f} above upper band ${context['upper_band']:.2f} (dev: {upper_deviation:.2f}%)"
            target_price = current_price - (context['atr'] * effective_target_mult)
            stop_loss = current_price + (context['atr'] * effective_stop_mult)

        else:
            if not volatility_ok:
                reason = f"Volatility {context['volatility_24h']:.2f}% > max {effective_max_volatility:.2f}%"
            elif not volume_ok:
                reason = f"Volume ratio {context['volume_ratio']:.2f} < min {effective_min_volume:.2f}"
            elif lower_deviation > 0 and lower_deviation < effective_min_deviation:
                reason = f"Deviation {lower_deviation:.2f}% < min {effective_min_deviation:.2f}%"
            elif upper_deviation > 0 and upper_deviation < effective_min_deviation:
                reason = f"Deviation {upper_deviation:.2f}% < min {effective_min_deviation:.2f}%"
            else:
                reason = "Price within bands"

        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'market_regime': regime,
            'regime_note': regime_adj['regime_note'],
            **context,
            'deviation_percent': max(lower_deviation, upper_deviation)
        }

    def check_regime_suitability(self, regime: str, lower_dev: float, upper_dev: float) -> Tuple[bool, str]:
        """
        Check if current market regime is suitable for VWAP mean reversion strategy.
        Returns (is_suitable, reason).
        """
        dev = max(lower_dev, upper_dev)

        if regime == "STRONG_TREND":
            # In strong trends, require larger deviations (trend may continue)
            min_required = self.params.min_deviation_percent * 1.5
            if dev > min_required:
                return True, f"Strong trend but deviation {dev:.2f}% > {min_required:.2f}%"
            return False, f"Strong trend - need {min_required:.2f}% deviation, only {dev:.2f}%"

        elif regime == "CHOPPY_HIGH_VOL":
            # In choppy high vol, be more selective
            min_required = self.params.min_deviation_percent * 1.3
            if dev > min_required:
                return True, f"Choppy market but deviation {dev:.2f}% > {min_required:.2f}%"
            return False, f"Choppy high vol - need {min_required:.2f}% deviation, only {dev:.2f}%"

        elif regime == "RANGING_LOW_VOL":
            # Perfect for mean reversion - more permissive
            return True, f"Ranging market - ideal for mean reversion"

        else:  # MIXED
            return True, f"Mixed regime - standard parameters apply"

    def classify_strategy_type(self, strategy_name: str) -> str:
        """Classify strategy type based on name for regime-specific adjustments"""
        name_upper = strategy_name.upper()

        if any(x in name_upper for x in ['MEAN_REVERSION', 'VWAP', 'REVERSAL', 'MARKET_MAKER']):
            return "MEAN_REVERSION"
        elif any(x in name_upper for x in ['BREAKOUT', 'MOMENTUM', 'TREND']):
            return "BREAKOUT"
        elif any(x in name_upper for x in ['COINTEGRATION', 'PAIRS', 'CORRELATION']):
            return "PAIRS_TRADING"
        elif any(x in name_upper for x in ['NEURAL', 'ML', 'MACHINE_LEARNING', 'AI']):
            return "ML_BASED"
        else:
            return "UNKNOWN"

    def load_and_execute_strategy(self, strategy_file: Path, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Dynamically load and execute a strategy from a Python file.

        This allows running the actual strategy code (including improved versions)
        instead of just using adaptive parameters.

        Args:
            strategy_file: Path to the strategy Python file
            df: DataFrame with OHLCV data
            current_price: Current price of the asset

        Returns:
            Signal dictionary with trade details
        """
        try:
            # Import strategy module dynamically
            spec = importlib.util.spec_from_file_location("strategy_module", strategy_file)
            if spec is None or spec.loader is None:
                print(f"⚠️  Could not load spec for {strategy_file}")
                return self.generate_signal(df, current_price)

            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)

            # Try to find and instantiate the strategy class
            strategy_instance = None

            # Look for common strategy class names
            strategy_class_names = [
                'ImprovedStrategy',
                'VWAPStrategy',
                'TradingStrategy',
                'Strategy',
                'MeanReversionStrategy',
                'MarketMakerStrategy'
            ]

            for class_name in strategy_class_names:
                if hasattr(strategy_module, class_name):
                    strategy_class = getattr(strategy_module, class_name)
                    try:
                        strategy_instance = strategy_class()
                        break
                    except Exception:
                        continue

            # If no known class found, look for any class with generate_signal method
            if strategy_instance is None:
                for name in dir(strategy_module):
                    obj = getattr(strategy_module, name)
                    if isinstance(obj, type) and hasattr(obj, 'generate_signal'):
                        try:
                            strategy_instance = obj()
                            break
                        except Exception:
                            continue

            # If we found a strategy, use it
            if strategy_instance is not None and hasattr(strategy_instance, 'generate_signal'):
                try:
                    signal = strategy_instance.generate_signal(df, current_price)
                    # Ensure signal has required fields
                    if isinstance(signal, dict) and 'signal' in signal:
                        # Add context if missing
                        if 'vwap' not in signal:
                            context = self.calculate_market_context(df)
                            signal.update(context)
                        return signal
                except Exception as e:
                    print(f"⚠️  Strategy {strategy_file.stem} generate_signal failed: {e}")

            # Fallback to adaptive parameters
            return self.generate_signal(df, current_price)

        except Exception as e:
            print(f"❌ Failed to load strategy {strategy_file}: {e}")
            # Fallback to adaptive parameters
            return self.generate_signal(df, current_price)

# =============================================================================
# TRADE ANALYZER - Identifies Loss Patterns
# =============================================================================

class TradeAnalyzer:
    """Analyzes trades to identify loss patterns and suggest optimizations"""

    def __init__(self):
        self.trades: List[TradeContext] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}

    def add_trade(self, trade: TradeContext):
        """Add completed trade for analysis"""
        self.trades.append(trade)
        self._update_strategy_performance(trade)

    def _update_strategy_performance(self, trade: TradeContext):
        """Update performance metrics for strategy"""
        strategy_id = trade.strategy_id

        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = StrategyPerformance(strategy_id=strategy_id)

        perf = self.strategy_performance[strategy_id]
        perf.total_trades += 1
        perf.total_pnl += trade.pnl_usd

        if trade.pnl_usd > 0:
            perf.winning_trades += 1
            perf.consecutive_wins += 1
            perf.consecutive_losses = 0

            if trade.exit_reason.startswith("TARGET"):
                perf.target_hits += 1
        else:
            perf.losing_trades += 1
            perf.consecutive_losses += 1
            perf.consecutive_wins = 0

            if trade.exit_reason.startswith("STOP"):
                perf.stop_loss_hits += 1

            # Categorize loss pattern
            if trade.deviation_percent < 0.5:
                perf.losses_low_deviation += 1
            if trade.volatility_24h > 3.0:
                perf.losses_high_volatility += 1
            if trade.trend_direction == "UP" and trade.direction == "SELL":
                perf.losses_in_uptrend += 1
            if trade.trend_direction == "DOWN" and trade.direction == "BUY":
                perf.losses_in_downtrend += 1

        # Calculate win rate
        if perf.total_trades > 0:
            perf.win_rate = perf.winning_trades / perf.total_trades

        # Calculate profit factor
        wins = [t.pnl_usd for t in self.trades if t.strategy_id == strategy_id and t.pnl_usd > 0]
        losses = [abs(t.pnl_usd) for t in self.trades if t.strategy_id == strategy_id and t.pnl_usd < 0]

        if losses and sum(losses) > 0:
            perf.profit_factor = sum(wins) / sum(losses) if wins else 0

        perf.avg_win = np.mean(wins) if wins else 0
        perf.avg_loss = np.mean(losses) if losses else 0

        # Average holding time
        holding_times = [t.holding_time_minutes for t in self.trades if t.strategy_id == strategy_id]
        perf.avg_holding_time = np.mean(holding_times) if holding_times else 0

        # Pause strategy if too many consecutive losses
        if perf.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            perf.is_paused = True
            print(f"⏸️  Strategy {strategy_id} PAUSED after {perf.consecutive_losses} consecutive losses")

    def identify_loss_patterns(self, strategy_id: str) -> List[str]:
        """Identify dominant loss patterns for a strategy"""
        patterns = []

        if strategy_id not in self.strategy_performance:
            return patterns

        perf = self.strategy_performance[strategy_id]

        if perf.losing_trades < Config.MIN_TRADES_FOR_ANALYSIS:
            return patterns

        # Calculate pattern frequencies
        total_losses = perf.losing_trades

        if perf.losses_low_deviation / total_losses > 0.4:
            patterns.append("LOW_DEVIATION")

        if perf.losses_high_volatility / total_losses > 0.4:
            patterns.append("HIGH_VOLATILITY")

        if perf.stop_loss_hits / total_losses > 0.7:
            patterns.append("TIGHT_STOPS")

        if perf.target_hits / perf.total_trades < 0.2 and perf.total_trades >= 5:
            patterns.append("FAR_TARGETS")

        if (perf.losses_in_uptrend + perf.losses_in_downtrend) / total_losses > 0.5:
            patterns.append("WRONG_TREND")

        return patterns

    def get_optimization_suggestions(self, strategy_id: str) -> Dict:
        """Get specific optimization suggestions"""
        patterns = self.identify_loss_patterns(strategy_id)
        perf = self.strategy_performance.get(strategy_id)

        suggestions = {
            'patterns': patterns,
            'suggestions': []
        }

        if not perf or perf.total_trades < Config.MIN_TRADES_FOR_ANALYSIS:
            return suggestions

        if "LOW_DEVIATION" in patterns:
            suggestions['suggestions'].append("Increase minimum deviation threshold - entries too close to VWAP")

        if "HIGH_VOLATILITY" in patterns:
            suggestions['suggestions'].append("Reduce maximum volatility threshold - avoid volatile markets")

        if "TIGHT_STOPS" in patterns:
            suggestions['suggestions'].append("Widen stop loss multiplier - stops triggered too easily")

        if "FAR_TARGETS" in patterns:
            suggestions['suggestions'].append("Reduce take profit multiplier - targets too ambitious")

        if "WRONG_TREND" in patterns:
            suggestions['suggestions'].append("Add trend filter - avoid counter-trend trades")

        return suggestions

    def print_analysis(self, strategy_id: str = None):
        """Print analysis for strategy or all strategies"""
        strategies = [strategy_id] if strategy_id else list(self.strategy_performance.keys())

        print("\n" + "="*80)
        print("📊 TRADE ANALYSIS REPORT")
        print("="*80)

        for sid in strategies:
            if sid not in self.strategy_performance:
                continue

            perf = self.strategy_performance[sid]
            patterns = self.identify_loss_patterns(sid)

            print(f"\n🎯 Strategy: {sid[:50]}")
            print(f"   Total Trades: {perf.total_trades}")
            print(f"   Win Rate: {perf.win_rate:.1%}")
            print(f"   Profit Factor: {perf.profit_factor:.2f}")
            print(f"   Total PnL: ${perf.total_pnl:+.2f}")
            print(f"   Avg Win: ${perf.avg_win:.2f} | Avg Loss: ${perf.avg_loss:.2f}")
            print(f"   Consecutive Losses: {perf.consecutive_losses}")
            print(f"   Status: {'⏸️ PAUSED' if perf.is_paused else '✅ ACTIVE'}")

            if patterns:
                print(f"   📉 Loss Patterns: {', '.join(patterns)}")

            suggestions = self.get_optimization_suggestions(sid)
            if suggestions['suggestions']:
                print(f"   💡 Suggestions:")
                for s in suggestions['suggestions']:
                    print(f"      - {s}")

        print("="*80 + "\n")

# =============================================================================
# PAPER TRADING ENGINE WITH ADAPTIVE LEARNING
# =============================================================================

@dataclass
class AdaptivePaperTrade:
    """Paper trade with full context"""
    strategy_id: str
    symbol: str
    direction: str
    entry_price: float
    size: float
    target_price: float
    stop_loss: float
    entry_time: datetime
    status: str = "OPEN"
    exit_price: float = 0.0
    pnl: float = 0.0

    # Market context at entry
    vwap: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    atr: float = 0.0
    deviation_percent: float = 0.0
    volatility_24h: float = 0.0
    volume_ratio: float = 0.0
    trend_direction: str = ""
    market_regime: str = ""  # NEW: CHOPPY_HIGH_VOL, STRONG_TREND, RANGING_LOW_VOL, MIXED
    regime_note: str = ""    # NEW: Human-readable regime adjustment note

class AdaptivePaperTradingEngine:
    """Paper trading engine with trade analysis and adaptive learning"""

    def __init__(self):
        self.positions: Dict[str, AdaptivePaperTrade] = {}
        self.capital = Config.STARTING_CAPITAL
        self.trade_history: List[TradeContext] = []
        self.analyzer = TradeAnalyzer()

        # Strategy-specific adaptive parameters
        self.strategy_params: Dict[str, AdaptiveParameters] = {}

        # LLM-based optimizer (like APEX RBI)
        self.llm_optimizer = LLMStrategyOptimizer()

        # Strategy recoder for saving improved strategies
        self.strategy_recoder = StrategyRecoder()

        # Track strategy files for recoding
        self.strategy_files: Dict[str, Path] = {}

        # Load any previously saved state (CRITICAL for persistence)
        self.load_state()

    def set_strategy_file(self, strategy_id: str, file_path: Path):
        """Register the original strategy file for potential recoding"""
        self.strategy_files[strategy_id] = file_path

    def get_params(self, strategy_id: str) -> AdaptiveParameters:
        """Get or create adaptive parameters for strategy"""
        if strategy_id not in self.strategy_params:
            self.strategy_params[strategy_id] = AdaptiveParameters()
        return self.strategy_params[strategy_id]

    def can_open_position(self, strategy_id: str, symbol: str) -> Tuple[bool, str]:
        """Check if position can be opened"""
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]

        # Check if strategy is paused
        if strategy_id in self.analyzer.strategy_performance:
            if self.analyzer.strategy_performance[strategy_id].is_paused:
                return False, f"Strategy {strategy_id} is PAUSED due to consecutive losses"

        # Total position limit
        if len(open_positions) >= Config.MAX_TOTAL_POSITIONS:
            return False, f"Max total positions ({Config.MAX_TOTAL_POSITIONS}) reached"

        # Strategy position limit
        strategy_positions = [p for p in open_positions if p.strategy_id == strategy_id]
        if len(strategy_positions) >= Config.MAX_POSITIONS_PER_STRATEGY:
            return False, f"Strategy has max positions ({Config.MAX_POSITIONS_PER_STRATEGY})"

        # Token position limit
        token_positions = [p for p in open_positions if p.symbol == symbol]
        if len(token_positions) >= Config.MAX_POSITIONS_PER_TOKEN:
            return False, f"Token {symbol} has max positions ({Config.MAX_POSITIONS_PER_TOKEN})"

        # Existing position check
        existing = [p for p in open_positions if p.strategy_id == strategy_id and p.symbol == symbol]
        if existing:
            return False, f"Already have position on {symbol}"

        return True, "OK"

    def open_position(self, strategy_id: str, symbol: str, signal: Dict) -> Optional[str]:
        """Open position with full context including market regime"""
        if signal['signal'] == 'HOLD':
            return None

        can_open, reason = self.can_open_position(strategy_id, symbol)
        if not can_open:
            print(f"⏸️  BLOCKED: {strategy_id[:30]} {symbol} - {reason}")
            return None

        params = self.get_params(strategy_id)
        position_id = f"{strategy_id}_{symbol}_{datetime.now().strftime('%H%M%S')}"
        size_usd = self.capital * params.position_size_percent
        leverage_size = size_usd * Config.DEFAULT_LEVERAGE

        position = AdaptivePaperTrade(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=signal['signal'],
            entry_price=signal['current_price'],
            size=leverage_size,
            target_price=signal.get('target_price', 0),
            stop_loss=signal.get('stop_loss', 0),
            entry_time=datetime.now(),
            vwap=signal.get('vwap', 0),
            upper_band=signal.get('upper_band', 0),
            lower_band=signal.get('lower_band', 0),
            atr=signal.get('atr', 0),
            deviation_percent=signal.get('deviation_percent', 0),
            volatility_24h=signal.get('volatility_24h', 0),
            volume_ratio=signal.get('volume_ratio', 0),
            trend_direction=signal.get('trend', ''),
            market_regime=signal.get('market_regime', ''),  # NEW: Market regime
            regime_note=signal.get('regime_note', '')       # NEW: Regime note
        )

        self.positions[position_id] = position

        # Enhanced logging with regime info
        regime = signal.get('market_regime', 'UNKNOWN')
        regime_note = signal.get('regime_note', '')
        print(f"🎯 OPENED: {position_id[:50]}")
        print(f"   {signal['signal']} {symbol} @ ${signal['current_price']:.2f}")
        print(f"   Target: ${signal.get('target_price', 0):.2f} | Stop: ${signal.get('stop_loss', 0):.2f}")
        print(f"   Deviation: {signal.get('deviation_percent', 0):.2f}% | Volatility: {signal.get('volatility_24h', 0):.2f}%")
        print(f"   Regime: {regime} {regime_note}")

        return position_id

    def check_real_time_metrics(self):
        """
        Real-time performance monitoring and alerts.
        Triggers immediate warnings when strategy performance deteriorates.
        """
        alerts = []

        for strategy_id, perf in self.analyzer.strategy_performance.items():
            # Check for consecutive losses
            if perf.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
                alerts.append(f"🚨 ALERT: {strategy_id[:40]} has {perf.consecutive_losses} consecutive losses!")
                if not perf.is_paused:
                    perf.is_paused = True
                    alerts.append(f"   ⏸️  Strategy PAUSED - will resume after optimization")

            # Check recent performance (last 5 trades)
            recent_trades = [t for t in self.trade_history if t.strategy_id == strategy_id][-5:]
            if len(recent_trades) >= 3:
                recent_wins = sum(1 for t in recent_trades if t.pnl_usd > 0)
                recent_win_rate = recent_wins / len(recent_trades)
                recent_pnl = sum(t.pnl_usd for t in recent_trades)

                if recent_win_rate < Config.ALERT_RECENT_WIN_RATE:
                    alerts.append(f"🚨 ALERT: {strategy_id[:40]} recent win rate only {recent_win_rate:.1%}!")
                    alerts.append(f"   Recent PnL: ${recent_pnl:+.2f}")

                # Check if losing money rapidly (proportional to capital)
                loss_threshold = Config.ALERT_RAPID_LOSS_THRESHOLD
                if recent_pnl < -loss_threshold:
                    alerts.append(f"🚨 ALERT: {strategy_id[:40]} rapid loss: ${recent_pnl:.2f} in last {len(recent_trades)} trades")

        # Check regime-specific performance
        regime_stats = {}
        for trade in self.trade_history[-50:]:  # Last 50 trades
            regime = getattr(trade, 'market_regime', 'UNKNOWN')
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'losses': 0}
            if trade.pnl_usd > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1

        # Alert on consistently bad regimes
        for regime, stats in regime_stats.items():
            total = stats['wins'] + stats['losses']
            if total >= Config.MIN_TRADES_FOR_REGIME_ALERT:
                win_rate = stats['wins'] / total
                if win_rate < Config.ALERT_REGIME_WIN_RATE:
                    alerts.append(f"⚠️  WARNING: Regime {regime} only {win_rate:.1%} win rate ({stats['wins']}W/{stats['losses']}L)")

        # Print alerts
        if alerts:
            print("\n" + "=" * 60)
            print("📊 REAL-TIME PERFORMANCE ALERTS")
            print("=" * 60)
            for alert in alerts:
                print(alert)
            print("=" * 60 + "\n")

        return alerts

    def check_exits(self, current_prices: Dict, cycle: int = 0):
        """Check and execute exits
        
        🚨 NEW: PORTFOLIO TAKE PROFIT
        - Close ALL positions when total unrealized PnL hits threshold
        - This banks profits before they disappear!
        
        🎯 NEW: TIME-BASED EXIT
        - Close all positions every N cycles if profitable
        """
        
        # ==========================================================================
        # 🚨 PORTFOLIO TAKE PROFIT - Close all when we hit profit threshold!
        # ==========================================================================
        if getattr(Config, 'ENABLE_PORTFOLIO_TAKE_PROFIT', True):
            total_unrealized = 0.0
            open_positions = [(pid, pos) for pid, pos in self.positions.items() if pos.status == "OPEN"]
            
            for position_id, position in open_positions:
                current_price = current_prices.get(position.symbol)
                if not current_price:
                    continue
                    
                if position.direction == "BUY":
                    unrealized = (current_price - position.entry_price) / position.entry_price * position.size
                else:
                    unrealized = (position.entry_price - current_price) / position.entry_price * position.size
                total_unrealized += unrealized
            
            # Check TAKE PROFIT threshold
            take_profit_threshold = getattr(Config, 'PORTFOLIO_TAKE_PROFIT_THRESHOLD', 200.0)
            if total_unrealized >= take_profit_threshold:
                print(f"\n{'='*80}")
                print(f"💰💰💰 PORTFOLIO TAKE PROFIT TRIGGERED! 💰💰💰")
                print(f"   Unrealized: ${total_unrealized:.2f} >= ${take_profit_threshold:.2f}")
                print(f"   Closing ALL {len(open_positions)} positions to bank profit!")
                print(f"{'='*80}")
                
                for position_id, position in open_positions:
                    current_price = current_prices.get(position.symbol)
                    if current_price:
                        if position.direction == "BUY":
                            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                        else:
                            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
                        self.close_position(position_id, current_price, f"PORTFOLIO_TAKE_PROFIT ({pnl_pct:+.2f}%)")
                return  # Done checking exits
            
            # Check STOP LOSS threshold
            stop_loss_threshold = getattr(Config, 'PORTFOLIO_STOP_LOSS_THRESHOLD', -400.0)
            if total_unrealized <= stop_loss_threshold:
                print(f"\n{'='*80}")
                print(f"🛑 PORTFOLIO STOP LOSS TRIGGERED!")
                print(f"   Unrealized: ${total_unrealized:.2f} <= ${stop_loss_threshold:.2f}")
                print(f"   Closing ALL positions to limit losses!")
                print(f"{'='*80}")
                
                for position_id, position in open_positions:
                    current_price = current_prices.get(position.symbol)
                    if current_price:
                        if position.direction == "BUY":
                            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                        else:
                            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
                        self.close_position(position_id, current_price, f"PORTFOLIO_STOP_LOSS ({pnl_pct:+.2f}%)")
                return
        
        # ==========================================================================
        # 🎯 TIME-BASED EXIT - Close all every N cycles if profitable
        # ==========================================================================
        if getattr(Config, 'ENABLE_TIME_BASED_EXIT', True) and cycle > 0:
            exit_cycles = getattr(Config, 'TIME_BASED_EXIT_CYCLES', 50)
            min_profit = getattr(Config, 'MIN_PROFIT_FOR_TIME_EXIT', 50.0)
            
            if cycle % exit_cycles == 0:
                total_unrealized = 0.0
                open_positions = [(pid, pos) for pid, pos in self.positions.items() if pos.status == "OPEN"]
                
                for position_id, position in open_positions:
                    current_price = current_prices.get(position.symbol)
                    if not current_price:
                        continue
                    if position.direction == "BUY":
                        unrealized = (current_price - position.entry_price) / position.entry_price * position.size
                    else:
                        unrealized = (position.entry_price - current_price) / position.entry_price * position.size
                    total_unrealized += unrealized
                
                if total_unrealized >= min_profit:
                    print(f"\n{'='*80}")
                    print(f"⏰ TIME-BASED EXIT - Cycle {cycle}")
                    print(f"   Unrealized: ${total_unrealized:.2f} >= ${min_profit:.2f}")
                    print(f"   Closing {len(open_positions)} positions")
                    print(f"{'='*80}")
                    
                    for position_id, position in open_positions:
                        current_price = current_prices.get(position.symbol)
                        if current_price:
                            if position.direction == "BUY":
                                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                            else:
                                pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
                            self.close_position(position_id, current_price, f"TIME_EXIT_CYCLE_{cycle} ({pnl_pct:+.2f}%)")
                    return
        
        # ==========================================================================
        # NORMAL EXIT CHECKING (individual positions)
        # ==========================================================================
        for position_id, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue

            current_price = current_prices.get(position.symbol)
            if not current_price:
                continue

            # Calculate PnL
            if position.direction == "BUY":
                pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_percent = (position.entry_price - current_price) / position.entry_price * 100

            exit_reason = None

            # Check target
            if position.direction == "BUY" and current_price >= position.target_price:
                exit_reason = f"TARGET_HIT (+{pnl_percent:.2f}%)"
            elif position.direction == "SELL" and current_price <= position.target_price:
                exit_reason = f"TARGET_HIT (+{pnl_percent:.2f}%)"

            # Check stop loss
            elif position.direction == "BUY" and current_price <= position.stop_loss:
                exit_reason = f"STOP_LOSS ({pnl_percent:.2f}%)"
            elif position.direction == "SELL" and current_price >= position.stop_loss:
                exit_reason = f"STOP_LOSS ({pnl_percent:.2f}%)"

            # Check max holding time
            params = self.get_params(position.strategy_id)
            holding_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
            if holding_minutes > params.max_holding_periods * 15:  # 15 min per period
                exit_reason = f"MAX_TIME ({pnl_percent:.2f}%)"

            if exit_reason:
                self.close_position(position_id, current_price, exit_reason)

    def close_position(self, position_id: str, exit_price: float, reason: str):
        """Close position and log for analysis"""
        position = self.positions[position_id]

        if position.direction == "BUY":
            pnl_percent = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_percent = (position.entry_price - exit_price) / position.entry_price

        pnl_usd = position.size * pnl_percent
        holding_minutes = (datetime.now() - position.entry_time).total_seconds() / 60

        position.status = "CLOSED"
        position.exit_price = exit_price
        position.pnl = pnl_usd
        self.capital += pnl_usd

        # Create trade context for analysis
        trade_context = TradeContext(
            timestamp=datetime.now().isoformat(),
            position_id=position_id,
            strategy_id=position.strategy_id,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            target_price=position.target_price,
            stop_loss=position.stop_loss,
            size_usd=position.size,
            pnl_usd=pnl_usd,
            pnl_percent=pnl_percent * 100,
            status="CLOSED",
            exit_reason=reason,
            vwap=position.vwap,
            upper_band=position.upper_band,
            lower_band=position.lower_band,
            atr=position.atr,
            deviation_percent=position.deviation_percent,
            volatility_24h=position.volatility_24h,
            volume_ratio=position.volume_ratio,
            trend_direction=position.trend_direction,
            holding_time_minutes=holding_minutes
        )

        self.trade_history.append(trade_context)
        self.analyzer.add_trade(trade_context)

        print(f"🔒 CLOSED: {position_id[:50]}")
        print(f"   PnL: ${pnl_usd:+.2f} ({pnl_percent*100:+.2f}%) - {reason}")

    def optimize_strategy(self, strategy_id: str):
        """
        Analyze and optimize strategy parameters using LLM reasoning.
        This is the core of the adaptive system - like APEX RBI but for live trading.
        """
        if strategy_id not in self.analyzer.strategy_performance:
            return

        perf = self.analyzer.strategy_performance[strategy_id]

        if perf.total_trades < Config.MIN_TRADES_FOR_ANALYSIS:
            print(f"⏳ {strategy_id[:30]}: Only {perf.total_trades} trades, need {Config.MIN_TRADES_FOR_ANALYSIS} for analysis")
            return

        # Check if optimization needed
        needs_optimization = (
            perf.win_rate < Config.TARGET_WIN_RATE or
            perf.profit_factor < Config.TARGET_PROFIT_FACTOR
        )

        if not needs_optimization:
            print(f"✅ {strategy_id[:30]}: Performance OK (WR: {perf.win_rate:.0%}, PF: {perf.profit_factor:.2f})")
            return

        # Check if we've exceeded max optimization iterations
        current_params = self.get_params(strategy_id)
        if current_params.optimization_count >= Config.MAX_OPTIMIZATION_ITERATIONS:
            print(f"⚠️  {strategy_id[:30]}: Max optimizations ({Config.MAX_OPTIMIZATION_ITERATIONS}) reached")
            return

        print(f"\n{'='*60}")
        print(f"🔧 LLM OPTIMIZATION: {strategy_id[:50]}")
        print(f"{'='*60}")
        print(f"   Current Win Rate: {perf.win_rate:.1%} (target: {Config.TARGET_WIN_RATE:.0%})")
        print(f"   Current Profit Factor: {perf.profit_factor:.2f} (target: {Config.TARGET_PROFIT_FACTOR})")
        print(f"   Total PnL: ${perf.total_pnl:.2f}")
        print(f"   Optimization #: {current_params.optimization_count + 1}")

        # Get recent trades for analysis
        recent_trades = [t for t in self.trade_history if t.strategy_id == strategy_id][-20:]

        # Get original strategy file path for potential recoding
        original_file = self.strategy_files.get(strategy_id, Path("unknown"))
        original_code = ""
        if original_file.exists():
            try:
                with open(original_file, 'r') as f:
                    original_code = f.read()
            except:
                pass

        # USE LLM TO ANALYZE AND OPTIMIZE (the key innovation!)
        new_params, new_code = self.llm_optimizer.analyze_and_optimize(
            strategy_id=strategy_id,
            params=current_params,
            performance=perf,
            recent_trades=recent_trades,
            original_strategy_code=original_code
        )

        if new_params:
            # Apply the new parameters
            self.strategy_params[strategy_id] = new_params

            # Save the improved strategy as a new file
            if original_file.exists():
                saved_path = self.strategy_recoder.recode_strategy(
                    strategy_id=strategy_id,
                    params=new_params,
                    performance=perf,
                    original_file=original_file
                )
                if saved_path:
                    print(f"📁 Improved strategy saved to: {saved_path}")

            # Reset consecutive losses to unpause strategy
            if perf.is_paused:
                perf.consecutive_losses = 0
                perf.is_paused = False
                print(f"   ✅ Strategy UNPAUSED after LLM optimization")

            print(f"\n📊 NEW PARAMETERS (v{new_params.version}):")
            print(f"   min_deviation: {new_params.min_deviation_percent:.2f}%")
            print(f"   max_volatility: {new_params.max_volatility_percent:.2f}%")
            print(f"   stop_loss_mult: {new_params.stop_loss_atr_multiplier:.2f}")
            print(f"   take_profit_mult: {new_params.take_profit_atr_multiplier:.2f}")
            print(f"   std_dev_mult: {new_params.std_dev_multiplier:.2f}")
        else:
            print(f"⚠️  Optimization failed, keeping current parameters")

        print(f"{'='*60}\n")

    def save_state(self):
        """Save trade history and parameters to files for persistence"""
        # Save trades
        trades_data = [t.to_dict() for t in self.trade_history]
        with open(Config.TRADE_LOG_FILE, 'w') as f:
            json.dump(trades_data, f, indent=2, default=str)

        # Save parameters (CRITICAL: This persists learned optimizations)
        params_data = {k: v.to_dict() for k, v in self.strategy_params.items()}
        with open(Config.PARAMETER_LOG_FILE, 'w') as f:
            json.dump(params_data, f, indent=2)

        # Save strategy performance metrics
        perf_data = {}
        for strategy_id, perf in self.analyzer.strategy_performance.items():
            perf_data[strategy_id] = {
                'total_trades': perf.total_trades,
                'winning_trades': perf.winning_trades,
                'losing_trades': perf.losing_trades,
                'total_pnl': perf.total_pnl,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'consecutive_losses': perf.consecutive_losses,
                'is_paused': perf.is_paused,
                'stop_loss_hits': perf.stop_loss_hits,
                'target_hits': perf.target_hits,
                'losses_low_deviation': perf.losses_low_deviation,
                'losses_high_volatility': perf.losses_high_volatility
            }
        with open(Config.ANALYSIS_LOG_FILE, 'w') as f:
            json.dump(perf_data, f, indent=2)

        print(f"💾 State saved: {len(self.trade_history)} trades, {len(self.strategy_params)} strategy params")

    def load_state(self):
        """Load saved state from previous sessions - CRITICAL for learning persistence"""
        loaded_params = False
        loaded_perf = False
        loaded_trades = False

        # Load saved parameters (learned optimizations)
        if Config.PARAMETER_LOG_FILE.exists():
            try:
                with open(Config.PARAMETER_LOG_FILE, 'r') as f:
                    params_data = json.load(f)
                for strategy_id, params_dict in params_data.items():
                    self.strategy_params[strategy_id] = AdaptiveParameters(**params_dict)
                loaded_params = True
                print(f"📂 Loaded {len(params_data)} saved parameter sets")
            except Exception as e:
                print(f"⚠️  Could not load parameters: {e}")

        # Load strategy performance metrics
        if Config.ANALYSIS_LOG_FILE.exists():
            try:
                with open(Config.ANALYSIS_LOG_FILE, 'r') as f:
                    perf_data = json.load(f)
                for strategy_id, metrics in perf_data.items():
                    perf = StrategyPerformance(strategy_id=strategy_id)
                    perf.total_trades = metrics.get('total_trades', 0)
                    perf.winning_trades = metrics.get('winning_trades', 0)
                    perf.losing_trades = metrics.get('losing_trades', 0)
                    perf.total_pnl = metrics.get('total_pnl', 0.0)
                    perf.win_rate = metrics.get('win_rate', 0.0)
                    perf.profit_factor = metrics.get('profit_factor', 0.0)
                    perf.consecutive_losses = metrics.get('consecutive_losses', 0)
                    perf.is_paused = metrics.get('is_paused', False)
                    perf.stop_loss_hits = metrics.get('stop_loss_hits', 0)
                    perf.target_hits = metrics.get('target_hits', 0)
                    perf.losses_low_deviation = metrics.get('losses_low_deviation', 0)
                    perf.losses_high_volatility = metrics.get('losses_high_volatility', 0)
                    self.analyzer.strategy_performance[strategy_id] = perf
                loaded_perf = True
                print(f"📂 Loaded performance data for {len(perf_data)} strategies")
            except Exception as e:
                print(f"⚠️  Could not load performance data: {e}")

        # Load trade history
        if Config.TRADE_LOG_FILE.exists():
            try:
                with open(Config.TRADE_LOG_FILE, 'r') as f:
                    trades_data = json.load(f)
                # Just count, don't reload into active memory
                loaded_trades = True
                print(f"📂 Found {len(trades_data)} historical trades on disk")
            except Exception as e:
                print(f"⚠️  Could not load trade history: {e}")

        if loaded_params or loaded_perf:
            print("✅ Previous learning state restored - continuing optimization from where we left off")
        else:
            print("🆕 Starting fresh - no previous state found")

# =============================================================================
# MAIN ADAPTIVE TRADING ENGINE
# =============================================================================

class AdaptiveTradingEngine:
    """Main engine with adaptive learning"""

    def __init__(self):
        self.htx_client = RealHTXClient()
        self.paper_engine = AdaptivePaperTradingEngine()
        self.strategies = self.load_strategies()
        self.cycle_count = 0
        self.start_time = datetime.now()

        # Create LLM signal generator for strategy-aware signals
        self.llm_signal_generator = LLMSignalGenerator(self.paper_engine.llm_optimizer)

        # Create strategy executors with adaptive parameters
        self.executors: Dict[str, AdaptiveStrategyExecutor] = {}
        for strategy_id, strategy_info in self.strategies.items():
            params = self.paper_engine.get_params(strategy_id)
            executor = AdaptiveStrategyExecutor(params)
            executor.strategy_file = strategy_info['py_file']  # Set strategy file for LLM analysis
            self.executors[strategy_id] = executor

            # Analyze strategy type using LLM
            strategy_logic = self.llm_signal_generator.parse_strategy_logic(strategy_info['py_file'])
            executor.strategy_type = strategy_logic.get('strategy_type', 'UNKNOWN')
            print(f"   📊 {strategy_id[:40]}: Type={executor.strategy_type}")

    def load_strategies(self) -> Dict:
        """
        Load strategies from directory.
        If USE_IMPROVED_STRATEGIES is True, will also check improved_strategies/ folder
        and use the LATEST version (v1, v2, v3...) of each strategy automatically.
        """
        strategies = {}

        print(f"🔍 Looking for strategies in: {Config.STRATEGIES_DIR}")

        # First, load original strategies
        original_strategies = {}
        if Config.STRATEGIES_DIR.exists():
            py_files = [f for f in Config.STRATEGIES_DIR.glob("*.py") if '_meta' not in str(f)]
            print(f"📁 Found {len(py_files)} original strategy files")

            for py_file in py_files:
                strategy_id = py_file.stem
                original_strategies[strategy_id] = py_file

        # Check for improved strategies if enabled
        improved_versions = {}  # Maps original_strategy_id -> (version, filepath)

        if Config.USE_IMPROVED_STRATEGIES and Config.IMPROVED_STRATEGIES_DIR.exists():
            # Look for BOTH naming conventions:
            # 1. Strategy_improved_v1.py (old naming)
            # 2. Strategy_v1.py (new naming - what the server has!)
            improved_files = list(Config.IMPROVED_STRATEGIES_DIR.glob("*_v*.py"))
            if improved_files:
                print(f"📁 Found {len(improved_files)} improved strategy files in improved_strategies/")

                for improved_file in improved_files:
                    filename = improved_file.stem
                    
                    # Try new naming: Strategy_v23.py
                    if "_v" in filename and "_improved_v" not in filename:
                        # Parse: 20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy_v23
                        parts = filename.rsplit("_v", 1)
                        if len(parts) == 2:
                            original_id = parts[0]
                            try:
                                version = int(parts[1])
                                # Keep track of highest version for each original strategy
                                if original_id not in improved_versions or version > improved_versions[original_id][0]:
                                    improved_versions[original_id] = (version, improved_file)
                                    print(f"   📌 {original_id[:50]}... → v{version}")
                            except ValueError:
                                continue
                    
                    # Also try old naming: Strategy_improved_v1.py
                    elif "_improved_v" in filename:
                        parts = filename.rsplit("_improved_v", 1)
                        if len(parts) == 2:
                            original_id = parts[0]
                            try:
                                version = int(parts[1])
                                if original_id not in improved_versions or version > improved_versions[original_id][0]:
                                    improved_versions[original_id] = (version, improved_file)
                                    print(f"   📌 {original_id[:50]}... → v{version}")
                            except ValueError:
                                continue

        # Build final strategy list, using improved versions where available
        for strategy_id, py_file in original_strategies.items():
            if strategy_id in improved_versions and Config.USE_IMPROVED_STRATEGIES:
                version, improved_file = improved_versions[strategy_id]
                strategies[strategy_id] = {
                    'py_file': improved_file,
                    'original_file': py_file,
                    'version': version,
                    'is_improved': True
                }
                # Register the ORIGINAL file for future recoding
                self.paper_engine.set_strategy_file(strategy_id, py_file)
                print(f"✅ LOADED: {strategy_id} (USING IMPROVED v{version})")
            else:
                strategies[strategy_id] = {
                    'py_file': py_file,
                    'original_file': py_file,
                    'version': 0,
                    'is_improved': False
                }
                self.paper_engine.set_strategy_file(strategy_id, py_file)
                print(f"✅ LOADED: {strategy_id}")

        # Summary
        improved_count = sum(1 for s in strategies.values() if s.get('is_improved', False))
        print(f"🎯 Total strategies: {len(strategies)} ({improved_count} using improved versions)")
        
        # Show strategy types
        for sid, sinfo in list(strategies.items())[:5]:
            stype = self.paper_engine.detect_strategy_type(sid)
            print(f"   📊 {sid[:40]}: Type={stype}")
        if len(strategies) > 5:
            print(f"   ... and {len(strategies) - 5} more")
        
        print()
        print("🚀" + "="*78)
        print("   STARTING LIVE TRADING LOOP - Watching for signals...")
        print("   • Each cycle: Fetch prices → Check signals → Open/Close trades")
        print("   • Trades will show with LIVE P&L when positions are open")
        print("   • Portfolio take profit at $200+ unrealized profit")
        print("   • Press Ctrl+C to stop and save state")
        print("="*80)
        print()

        return strategies

    def refresh_executor(self, strategy_id: str):
        """
        Refresh a strategy's executor after optimization.
        This ensures the new parameters are used for signal generation.
        Called automatically after LLM optimization creates a new version.
        """
        if strategy_id in self.strategies:
            # Get the updated parameters
            params = self.paper_engine.get_params(strategy_id)
            # Create a new executor with the updated parameters
            self.executors[strategy_id] = AdaptiveStrategyExecutor(params)

            # Check if there's a new improved version available
            self._check_for_improved_version(strategy_id)

            print(f"🔄 Executor refreshed for: {strategy_id[:40]} (v{params.version})")

    def _check_for_improved_version(self, strategy_id: str):
        """Check if a new improved version exists and update strategy info"""
        if not Config.USE_IMPROVED_STRATEGIES or not Config.IMPROVED_STRATEGIES_DIR.exists():
            return

        # Look for improved versions - BOTH naming conventions!
        # 1. Strategy_v23.py (new naming - what server has)
        # 2. Strategy_improved_v23.py (old naming)
        pattern1 = f"{strategy_id}_v*.py"
        pattern2 = f"{strategy_id}_improved_v*.py"
        improved_files = list(Config.IMPROVED_STRATEGIES_DIR.glob(pattern1))
        improved_files.extend(list(Config.IMPROVED_STRATEGIES_DIR.glob(pattern2)))

        if improved_files:
            # Find the highest version
            best_version = 0
            best_file = None

            for f in improved_files:
                try:
                    version = int(f.stem.rsplit("_v", 1)[1])
                    if version > best_version:
                        best_version = version
                        best_file = f
                except (ValueError, IndexError):
                    continue

            if best_file and strategy_id in self.strategies:
                current_version = self.strategies[strategy_id].get('version', 0)
                if best_version > current_version:
                    self.strategies[strategy_id]['py_file'] = best_file
                    self.strategies[strategy_id]['version'] = best_version
                    self.strategies[strategy_id]['is_improved'] = True
                    print(f"🆕 Now using improved v{best_version} for: {strategy_id[:40]}")

    def execute_strategy_for_token(self, strategy_id: str, token: str):
        """Execute strategy with LLM-based signal generation based on actual strategy code"""
        try:
            df = self.htx_client.fetch_candles(token, '15min', 100)
            if df is None or len(df) < 50:
                return

            current_price = self.htx_client.get_current_price(token)
            if not current_price:
                return

            # Get executor and strategy info
            executor = self.executors[strategy_id]
            strategy_info = self.strategies.get(strategy_id, {})
            strategy_file = strategy_info.get('py_file')

            # Use LLM signal generator to generate signals based on strategy type
            if strategy_file and strategy_file.exists():
                signal = self.llm_signal_generator.generate_signal_for_strategy(
                    strategy_file, df, current_price, executor.params
                )
            else:
                # Fallback to generic signal generation
                signal = executor.generate_signal(df, current_price)

            if signal['signal'] != 'HOLD':
                strategy_type = executor.strategy_type if hasattr(executor, 'strategy_type') else 'UNKNOWN'
                print(f"\n🚀 SIGNAL: {strategy_id[:40]} - {signal['signal']} {token}")
                print(f"   Type: {strategy_type} | Reason: {signal['reason']}")
                self.paper_engine.open_position(strategy_id, token, signal)

        except Exception as e:
            print(f"❌ Error: {strategy_id[:30]} {token}: {e}")

    def display_status(self):
        """Display current status"""
        open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
        closed_positions = [p for p in self.paper_engine.positions.values() if p.status == "CLOSED"]
        total_pnl = sum(p.pnl for p in closed_positions)
        total_closed_trades = len(self.paper_engine.trade_history)

        print(f"\n{'='*80}")
        print(f"🔄 TRADEPEX ADAPTIVE - Dynamic Strategy Optimizer")
        print(f"{'='*80}")
        print(f"⏰ Cycle: {self.cycle_count} | Runtime: {datetime.now() - self.start_time}")
        print(f"💰 Capital: ${self.paper_engine.capital:.2f} | Total PnL: ${total_pnl:+.2f}")
        print(f"📊 Open: {len(open_positions)}/{Config.MAX_TOTAL_POSITIONS} | Closed: {total_closed_trades}")

        # Show optimization status
        if total_closed_trades < Config.MIN_TRADES_FOR_ANALYSIS:
            print(f"🔄 Optimization: WAITING ({total_closed_trades}/{Config.MIN_TRADES_FOR_ANALYSIS} closed trades needed)")
        else:
            print(f"🔄 Optimization: ACTIVE (analyzing {total_closed_trades} trades)")

        # Performance summary
        if total_closed_trades > 0:
            wins = len([t for t in self.paper_engine.trade_history if t.pnl_usd > 0])
            win_rate = wins / total_closed_trades * 100
            print(f"📈 Win Rate: {win_rate:.1f}% ({wins}/{total_closed_trades})")

        if open_positions:
            print(f"\n📊 OPEN POSITIONS (waiting to close for analysis):")
            total_unrealized_pnl = 0.0
            
            for position in open_positions:
                current_price = self.htx_client.get_current_price(position.symbol) or position.entry_price
                if position.direction == "BUY":
                    pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
                    pnl_usd = (current_price - position.entry_price) / position.entry_price * position.size * Config.DEFAULT_LEVERAGE
                    dist_to_target = (position.target_price - current_price) / current_price * 100
                    dist_to_stop = (current_price - position.stop_loss) / current_price * 100
                else:
                    pnl_percent = (position.entry_price - current_price) / position.entry_price * 100
                    pnl_usd = (position.entry_price - current_price) / position.entry_price * position.size * Config.DEFAULT_LEVERAGE
                    dist_to_target = (current_price - position.target_price) / current_price * 100
                    dist_to_stop = (position.stop_loss - current_price) / current_price * 100
                
                total_unrealized_pnl += pnl_usd

                # Calculate time in trade
                time_in_trade = (datetime.now() - position.entry_time).total_seconds() / 60

                # Full details with prices and P&L in dollars
                emoji = "🟢" if pnl_percent > 0 else "🔴"
                print(f"   📍 {position.strategy_id[:35]}")
                print(f"      {position.direction} {position.symbol} @ ${position.entry_price:.2f} → Now: ${current_price:.2f} ({pnl_percent:+.2f}%) {emoji} ${pnl_usd:+.2f}")
                print(f"      🎯 Target: ${position.target_price:.2f} ({dist_to_target:+.2f}% away) | 🛑 Stop: ${position.stop_loss:.2f} ({dist_to_stop:.2f}% away)")
                print(f"      ⏱️ Time: {time_in_trade:.0f}min | Dev: {position.deviation_percent:.2f}% | Vol: {position.volatility_24h:.2f}%")
            
            # Show total portfolio unrealized P&L
            total_emoji = "🟢" if total_unrealized_pnl > 0 else "🔴"
            print(f"\n   {total_emoji} TOTAL PORTFOLIO UNREALIZED P&L: ${total_unrealized_pnl:+.2f}")
            
            # Check if we're close to take profit threshold
            if Config.ENABLE_PORTFOLIO_TAKE_PROFIT:
                if total_unrealized_pnl >= Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD:
                    print(f"   💰 TAKE PROFIT THRESHOLD HIT! (${total_unrealized_pnl:+.2f} >= ${Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD})")
                elif total_unrealized_pnl > 0:
                    distance = Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD - total_unrealized_pnl
                    print(f"   📈 ${distance:.2f} away from take profit threshold (${Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD})")

        print(f"{'='*80}\n")

    def run_adaptive(self):
        """Run with adaptive learning and real-time monitoring"""
        print("🚀 STARTING TRADEPEX ADAPTIVE - Dynamic Strategy Optimizer")
        print(f"🎯 Strategies: {len(self.strategies)}")
        print(f"💰 Tokens: {Config.TRADEABLE_TOKENS}")
        print(f"⏰ Check interval: {Config.CHECK_INTERVAL}s")
        print(f"🔧 Optimization interval: Every {Config.OPTIMIZATION_INTERVAL} cycles")
        print("="*80)

        while True:
            self.cycle_count += 1

            try:
                print(f"\n🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Get current prices
                current_prices = {}
                for token in Config.TRADEABLE_TOKENS:
                    price = self.htx_client.get_current_price(token)
                    if price:
                        current_prices[token] = price
                        print(f"   {token}: ${price:.2f}")

                # Check exits first
                self.paper_engine.check_exits(current_prices)

                # NEW: Real-time performance monitoring
                self.paper_engine.check_real_time_metrics()

                # Look for new opportunities - iterate TOKEN first, then strategies
                # This ensures we spread positions across different tokens
                open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
                position_limit_reached = len(open_positions) >= Config.MAX_TOTAL_POSITIONS
                
                # Show live P&L for open positions
                if open_positions:
                    total_unrealized = 0
                    print(f"\n📊 OPEN POSITIONS ({len(open_positions)}):")
                    for pos in open_positions:
                        curr_price = current_prices.get(pos.symbol, pos.entry_price)
                        if pos.direction == "BUY":
                            pnl_pct = (curr_price - pos.entry_price) / pos.entry_price * 100
                            pnl_usd = (curr_price - pos.entry_price) / pos.entry_price * pos.size * Config.DEFAULT_LEVERAGE
                        else:
                            pnl_pct = (pos.entry_price - curr_price) / pos.entry_price * 100
                            pnl_usd = (pos.entry_price - curr_price) / pos.entry_price * pos.size * Config.DEFAULT_LEVERAGE
                        total_unrealized += pnl_usd
                        emoji = "🟢" if pnl_pct > 0 else "🔴"
                        print(f"   {emoji} {pos.direction} {pos.symbol} @ ${pos.entry_price:.2f} → ${curr_price:.2f} ({pnl_pct:+.2f}%) = ${pnl_usd:+.2f}")
                    
                    total_emoji = "🟢" if total_unrealized > 0 else "🔴"
                    print(f"   {total_emoji} TOTAL UNREALIZED P&L: ${total_unrealized:+.2f}")
                    
                    # Show distance to take profit threshold
                    if Config.ENABLE_PORTFOLIO_TAKE_PROFIT:
                        if total_unrealized >= Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD:
                            print(f"   💰 TAKE PROFIT THRESHOLD REACHED! (${total_unrealized:+.2f} >= ${Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD})")
                        elif total_unrealized > 0:
                            distance = Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD - total_unrealized
                            print(f"   📈 ${distance:.2f} away from take profit threshold (${Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD})")
                        else:
                            # When in loss, actual distance = get to breakeven + reach threshold
                            distance = abs(total_unrealized) + Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD
                            print(f"   📉 ${distance:.2f} away from take profit threshold (need +${abs(total_unrealized):.2f} to breakeven, then +${Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD:.2f} more)")
                    
                    # Check portfolio take profit
                    if Config.ENABLE_PORTFOLIO_TAKE_PROFIT and total_unrealized >= Config.PORTFOLIO_TAKE_PROFIT_THRESHOLD:
                        print(f"\n💰 PORTFOLIO TAKE PROFIT TRIGGERED! Closing all {len(open_positions)} positions for ${total_unrealized:+.2f}")
                else:
                    print(f"\n📊 No open positions - scanning for signals...")

                if not position_limit_reached:
                    for token in Config.TRADEABLE_TOKENS:
                        if position_limit_reached:
                            break  # Exit token loop if max positions reached

                        # Skip token if it already has max positions
                        token_positions = [p for p in open_positions if p.symbol == token]
                        if len(token_positions) >= Config.MAX_POSITIONS_PER_TOKEN:
                            continue  # Skip to next token silently

                        for strategy_id in self.strategies:
                            # Check position count before each trade attempt
                            current_open_count = len([p for p in self.paper_engine.positions.values() if p.status == "OPEN"])
                            if current_open_count >= Config.MAX_TOTAL_POSITIONS:
                                position_limit_reached = True
                                break  # Exit strategy loop

                            self.execute_strategy_for_token(strategy_id, token)
                            time.sleep(0.3)

                        # Refresh open_positions after processing each token
                        open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
                else:
                    print(f"⏸️  Position limit reached ({len(open_positions)}/{Config.MAX_TOTAL_POSITIONS})")

                # Periodic optimization
                if self.cycle_count % Config.OPTIMIZATION_INTERVAL == 0:
                    print("\n🔧 RUNNING OPTIMIZATION CYCLE...")

                    # Show current status
                    open_pos = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
                    closed_pos = [p for p in self.paper_engine.positions.values() if p.status == "CLOSED"]
                    total_closed_trades = len(self.paper_engine.trade_history)

                    print(f"   📊 Status: {len(open_pos)} open positions, {total_closed_trades} closed trades")
                    print(f"   📋 Need {Config.MIN_TRADES_FOR_ANALYSIS} closed trades per strategy for LLM analysis")

                    if total_closed_trades == 0:
                        print(f"   ⏳ WAITING: No trades closed yet - need trades to complete (hit target/stop/timeout)")
                        print(f"   💡 Trades will close when: target hit, stop loss hit, or ~3hr max hold timeout")
                    else:
                        # Run optimization for each strategy
                        for strategy_id in self.strategies:
                            self.paper_engine.optimize_strategy(strategy_id)
                            # Refresh executor if optimization happened
                            if strategy_id in self.paper_engine.strategy_params:
                                params = self.paper_engine.get_params(strategy_id)
                                if params.optimization_count > 0:
                                    self.refresh_executor(strategy_id)

                    # Print analysis
                    self.paper_engine.analyzer.print_analysis()

                    # Save state
                    self.paper_engine.save_state()

                self.display_status()
                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 SHUTDOWN REQUESTED...")
                self.paper_engine.save_state()
                self.paper_engine.analyzer.print_analysis()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("🎯 TRADEPEX ADAPTIVE - Dynamic Live Strategy Analyzer & Optimizer")
    print("="*80)
    print("Features:")
    print("  • Detailed trade context logging")
    print("  • Loss pattern recognition and analysis")
    print("  • Dynamic parameter adjustment")
    print("  • Automated optimization until profitability")
    print("="*80)

    engine = AdaptiveTradingEngine()
    engine.run_adaptive()
