#!/usr/bin/env python3
"""
🚀 TRADEPEX ADAPTIVE - THE ULTIMATE SELF-IMPROVING TRADING MACHINE
====================================================================

THIS IS NOT V1 - THIS IS THE EVOLUTION!

WHAT IT DOES:
1. TRADES strategies from successful_strategies/ folder
2. LOGS EVERYTHING - 100% full detailed logs like original TradePexV1
3. LEARNS from every trade - what works, what doesn't
4. GENERATES NEW STRATEGY VERSIONS (V1 → V2 → V3 → V4...)
5. TRADES the new versions until TARGET HIT!
6. ITERATES FOREVER - Always improving

THE IMPROVEMENT LOOP:
┌────────────────────────────────────────────────────────────────────┐
│  Strategy_V1.py → TRADE → LEARN → Generate Strategy_V2.py         │
│  Strategy_V2.py → TRADE → LEARN → Generate Strategy_V3.py         │
│  Strategy_V3.py → TRADE → LEARN → Generate Strategy_V4.py         │
│  ... UNTIL TARGET WIN RATE + PROFIT ACHIEVED ...                  │
└────────────────────────────────────────────────────────────────────┘

TARGET: 60% Win Rate + Positive PnL = CHAMPION STATUS

🌙 Trade → Learn → Code → Improve → Repeat FOREVER 🌙
"""

import os
import sys
import json
import time
import ast
import re
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import traceback

# Optional LLM imports (same pattern as apex.py and tradeadapt.py)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """TradePex Adaptive Configuration"""
    
    # Capital
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.10
    DEFAULT_LEVERAGE = 3
    
    # Exchange
    HTX_BASE_URL = "https://api.huobi.pro"
    
    # ALL THE TOKENS - MORE TRADING = MORE LEARNING
    TRADEABLE_TOKENS = [
        'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX',
        'MATIC', 'ATOM', 'NEAR', 'FTM', 'ALGO', 'UNI', 'AAVE',
        'OP', 'ARB', 'DOGE', 'SHIB', 'LTC', 'BCH', 'ETC', 'APT', 'SUI'
    ]
    
    # Directories - Support both local and server paths
    STRATEGIES_DIR = Path(__file__).parent / "successful_strategies"
    SERVER_STRATEGIES_DIR = Path("/root/KEEP_SAFE/v1/APEX/successful_strategies")
    GENERATED_DIR = Path(__file__).parent / "generated_strategies"
    IMPROVED_DIR = Path(__file__).parent / "improved_strategies"
    LOGS_DIR = Path(__file__).parent / "tradepex_adaptive_logs"
    LEARNING_FILE = Path(__file__).parent / "tradepex_learning.json"
    
    # Trading
    CHECK_INTERVAL = 20  # Seconds
    MAX_TOTAL_POSITIONS = 20
    MAX_POSITIONS_PER_STRATEGY = 5
    MAX_POSITIONS_PER_TOKEN = 4
    
    # TARGETS - When these are hit, strategy becomes CHAMPION and can GO LIVE!
    TARGET_WIN_RATE = 0.60  # 60%
    TARGET_PROFIT_FACTOR = 1.5
    MIN_TRADES_FOR_EVALUATION = 20
    MIN_TRADES_FOR_ANALYSIS = 5  # Min trades before LLM analysis
    
    # Iteration - Generate new version every N trades
    ITERATION_INTERVAL = 30
    OPTIMIZATION_INTERVAL = 10  # Run LLM optimization every N cycles
    MAX_OPTIMIZATION_ITERATIONS = 10  # Max times to optimize before giving up
    MAX_CONSECUTIVE_LOSSES = 5  # Pause strategy after this many consecutive losses
    
    # LLM Configuration for Reasoning-Based Optimization (like APEX RBI)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")


# =============================================================================
# DATA CLASSES FOR COMPLETE TRADE TRACKING (from tradeadapt.py)
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
    volume_ratio: float = 0.0
    trend_direction: str = ""
    market_regime: str = ""
    
    # Duration
    holding_time_minutes: float = 0.0
    holding_periods: int = 0
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'position_id': self.position_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'size_usd': self.size_usd,
            'pnl_usd': self.pnl_usd,
            'pnl_percent': self.pnl_percent,
            'status': self.status,
            'exit_reason': self.exit_reason,
            'vwap': self.vwap,
            'deviation_percent': self.deviation_percent,
            'volatility_24h': self.volatility_24h,
            'trend_direction': self.trend_direction,
            'market_regime': self.market_regime,
            'holding_time_minutes': self.holding_time_minutes
        }


@dataclass
class StrategyPerformance:
    """Detailed performance metrics for a strategy"""
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
    losses_low_deviation: int = 0
    losses_high_volatility: int = 0


@dataclass
class AdaptiveParameters:
    """Dynamically adjustable strategy parameters - learned from trading"""
    # Entry Conditions
    min_deviation_percent: float = 0.15
    max_volatility_percent: float = 8.0
    min_volume_ratio: float = 0.3
    
    # Exit Conditions
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    max_holding_periods: int = 12
    
    # Technical Parameters
    std_dev_multiplier: float = 2.0
    vwap_periods: int = 20
    atr_periods: int = 14
    
    # Optimization Tracking
    version: int = 1
    optimization_count: int = 0
    last_optimized: str = ""
    
    def to_dict(self) -> dict:
        return {
            'min_deviation_percent': self.min_deviation_percent,
            'max_volatility_percent': self.max_volatility_percent,
            'min_volume_ratio': self.min_volume_ratio,
            'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
            'take_profit_atr_multiplier': self.take_profit_atr_multiplier,
            'max_holding_periods': self.max_holding_periods,
            'std_dev_multiplier': self.std_dev_multiplier,
            'version': self.version,
            'optimization_count': self.optimization_count
        }


# =============================================================================
# TRADE ANALYZER - IDENTIFIES LOSS PATTERNS (from tradeadapt.py)
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
        
        total_losses = perf.losing_trades
        
        if total_losses > 0:
            if perf.losses_low_deviation / total_losses > 0.4:
                patterns.append("LOW_DEVIATION")
            if perf.losses_high_volatility / total_losses > 0.4:
                patterns.append("HIGH_VOLATILITY")
            if perf.stop_loss_hits / total_losses > 0.7:
                patterns.append("TIGHT_STOPS")
            if perf.total_trades >= 5 and perf.target_hits / perf.total_trades < 0.2:
                patterns.append("FAR_TARGETS")
            if (perf.losses_in_uptrend + perf.losses_in_downtrend) / total_losses > 0.5:
                patterns.append("WRONG_TREND")
        
        return patterns
    
    def get_optimization_suggestions(self, strategy_id: str) -> Dict:
        """Get specific optimization suggestions based on loss patterns"""
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
        """Print detailed analysis for strategy or all strategies"""
        strategies = [strategy_id] if strategy_id else list(self.strategy_performance.keys())
        
        print(f"\n{'='*100}")
        print(f"📊 TRADE ANALYSIS REPORT")
        print(f"{'='*100}")
        
        for sid in strategies:
            if sid not in self.strategy_performance:
                continue
            
            perf = self.strategy_performance[sid]
            patterns = self.identify_loss_patterns(sid)
            suggestions = self.get_optimization_suggestions(sid)
            
            print(f"\n📈 Strategy: {sid[:50]}")
            print(f"   Total Trades:      {perf.total_trades}")
            print(f"   Win Rate:          {perf.win_rate:.1%}")
            print(f"   Profit Factor:     {perf.profit_factor:.2f}")
            print(f"   Total PnL:         ${perf.total_pnl:+.2f}")
            print(f"   Avg Win:           ${perf.avg_win:.2f}")
            print(f"   Avg Loss:          ${perf.avg_loss:.2f}")
            print(f"   Consecutive Losses: {perf.consecutive_losses}")
            print(f"   Stop Hits:         {perf.stop_loss_hits}")
            print(f"   Target Hits:       {perf.target_hits}")
            
            if patterns:
                print(f"\n   🔍 LOSS PATTERNS IDENTIFIED:")
                for pattern in patterns:
                    print(f"      ⚠️  {pattern}")
            
            if suggestions['suggestions']:
                print(f"\n   💡 OPTIMIZATION SUGGESTIONS:")
                for suggestion in suggestions['suggestions']:
                    print(f"      → {suggestion}")
        
        print(f"\n{'='*100}")


# =============================================================================
# STRATEGY RECODER - GENERATES IMPROVED .PY FILES (from tradeadapt.py)
# =============================================================================

class StrategyRecoder:
    """
    Recodes and saves improved strategies as new Python files.
    - Original strategies in successful_strategies/ are NEVER modified
    - Improved versions are saved to improved_strategies/ folder
    - Each improvement creates a new version (v1, v2, v3, etc.)
    """
    
    def __init__(self):
        Config.IMPROVED_DIR.mkdir(parents=True, exist_ok=True)
        self.improvement_history: Dict[str, List[dict]] = {}
        self._load_history()
    
    def _load_history(self):
        """Load improvement history from file"""
        history_file = Config.IMPROVED_DIR / "improvement_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.improvement_history = json.load(f)
                print(f"📂 Loaded improvement history for {len(self.improvement_history)} strategies")
            except Exception as e:
                print(f"⚠️  Could not load improvement history: {e}")
    
    def _save_history(self):
        """Save improvement history to file"""
        history_file = Config.IMPROVED_DIR / "improvement_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.improvement_history, f, indent=2, default=str)
    
    def get_next_version(self, strategy_id: str) -> int:
        """Get the next version number for a strategy"""
        if strategy_id not in self.improvement_history:
            return 1
        return len(self.improvement_history[strategy_id]) + 1
    
    def recode_strategy(self, strategy_id: str, params: AdaptiveParameters,
                       performance: StrategyPerformance, original_file: Path) -> Optional[Path]:
        """
        Recode a strategy with improved parameters and save as new file.
        Returns the path to the new strategy file.
        """
        version = self.get_next_version(strategy_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_filename = f"{strategy_id}_improved_v{version}.py"
        new_filepath = Config.IMPROVED_DIR / new_filename
        
        # Generate the improved strategy code
        strategy_code = self._generate_improved_strategy_code(
            strategy_id=strategy_id,
            params=params,
            performance=performance,
            original_file=original_file,
            version=version
        )
        
        try:
            with open(new_filepath, 'w') as f:
                f.write(strategy_code)
            
            # Save metadata
            meta_filepath = Config.IMPROVED_DIR / f"{strategy_id}_improved_v{version}_meta.json"
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
                }
            }
            with open(meta_filepath, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            # Update history
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
                                         performance: StrategyPerformance,
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

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

# =============================================================================
# OPTIMIZED PARAMETERS (learned from live trading)
# =============================================================================

@dataclass
class OptimizedParameters:
    """Parameters optimized from {performance.total_trades} trades"""
    # Entry Conditions
    MIN_DEVIATION_PERCENT: float = {params.min_deviation_percent:.4f}
    MAX_VOLATILITY_PERCENT: float = {params.max_volatility_percent:.4f}
    MIN_VOLUME_RATIO: float = {params.min_volume_ratio:.4f}
    
    # Exit Conditions
    STOP_LOSS_ATR_MULTIPLIER: float = {params.stop_loss_atr_multiplier:.4f}
    TAKE_PROFIT_ATR_MULTIPLIER: float = {params.take_profit_atr_multiplier:.4f}
    MAX_HOLDING_PERIODS: int = {params.max_holding_periods}
    
    # Technical Parameters
    STD_DEV_MULTIPLIER: float = {params.std_dev_multiplier:.4f}
    VWAP_PERIODS: int = {params.vwap_periods}
    ATR_PERIODS: int = {params.atr_periods}


class ImprovedStrategy:
    """
    Improved VWAP Mean Reversion Strategy
    Version {version} - Optimized from live trading data
    """
    
    def __init__(self):
        self.params = OptimizedParameters()
    
    def calculate_vwap_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate VWAP with bands"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_vp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        vwap = cumulative_vp / cumulative_volume
        
        # Standard deviation bands
        squared_diff = ((typical_price - vwap) ** 2 * df['Volume']).cumsum()
        std_dev = np.sqrt(squared_diff / cumulative_volume)
        
        upper_band = vwap + (std_dev * self.params.STD_DEV_MULTIPLIER)
        lower_band = vwap - (std_dev * self.params.STD_DEV_MULTIPLIER)
        
        return {{
            'vwap': vwap.iloc[-1],
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1],
            'std_dev': std_dev.iloc[-1]
        }}
    
    def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close)
        ], axis=1).max(axis=1)
        
        return tr.rolling(self.params.ATR_PERIODS).mean().iloc[-1]
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Generate trading signal based on optimized parameters"""
        bands = self.calculate_vwap_bands(df)
        atr = self.calculate_atr(df)
        
        # Calculate deviations
        lower_deviation = (bands['vwap'] - current_price) / bands['vwap'] * 100
        upper_deviation = (current_price - bands['vwap']) / bands['vwap'] * 100
        
        # Calculate volatility (24h range)
        volatility = (df['High'].max() - df['Low'].min()) / df['Close'].mean() * 100
        volatility_ok = volatility <= self.params.MAX_VOLATILITY_PERCENT
        
        signal = "HOLD"
        reason = ""
        target_price = 0.0
        stop_loss = 0.0
        
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
# USAGE
# =============================================================================

if __name__ == "__main__":
    print("Improved Strategy: {strategy_id}")
    print(f"Version: {version}")
    
    strategy = ImprovedStrategy()
    print(f"Min deviation: {{strategy.params.MIN_DEVIATION_PERCENT:.4f}}%")
    print(f"Stop loss multiplier: {{strategy.params.STOP_LOSS_ATR_MULTIPLIER:.4f}}")
    print(f"Take profit multiplier: {{strategy.params.TAKE_PROFIT_ATR_MULTIPLIER:.4f}}")
'''
        return code


# =============================================================================
# FULL LOGGING SYSTEM - 100% DETAILED LOGS LIKE ORIGINAL V1
# =============================================================================

class FullLogger:
    """
    100% FULL LOGGING - Shows EVERYTHING like original TradePexV1!
    
    Logs:
    - Every price fetch
    - Every signal generated
    - Every position opened/closed
    - Every iteration
    - Every strategy version generated
    - Full performance stats
    """
    
    def __init__(self):
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = Config.LOGS_DIR / f"tradepex_adaptive_{timestamp}.log"
        self.trades_csv = Config.LOGS_DIR / f"trades_{timestamp}.csv"
        self.signals_csv = Config.LOGS_DIR / f"signals_{timestamp}.csv"
        self.versions_log = Config.LOGS_DIR / f"versions_{timestamp}.log"
        
        # Init CSV files
        with open(self.trades_csv, 'w') as f:
            f.write("timestamp,cycle,strategy,strategy_version,token,action,direction,price,size,target,stop,pnl,pnl_pct,reason\n")
        
        with open(self.signals_csv, 'w') as f:
            f.write("timestamp,cycle,strategy,token,signal,price,vwap,upper,lower,rsi,momentum,reason\n")
        
        # Stats
        self.cycle = 0
        self.total_signals = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0
        self.trades_opened = 0
        self.trades_closed = 0
        self.wins = 0
        self.losses = 0
    
    def _write_log(self, msg: str):
        """Write to log file"""
        with open(self.main_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")
    
    def log_cycle_start(self, cycle: int, prices: Dict[str, float]):
        """Log start of trading cycle"""
        self.cycle = cycle
        
        print(f"\n{'='*100}")
        print(f"🔄 CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        print(f"\n📊 LIVE PRICES FROM HTX:")
        
        for token, price in sorted(prices.items()):
            print(f"   {token:6} : ${price:>12,.4f}")
        
        self._write_log(f"CYCLE {cycle} START | Prices: {len(prices)}")
    
    def log_signal(self, strategy: str, version: int, token: str, signal_data: Dict):
        """Log trading signal with FULL details"""
        self.total_signals += 1
        
        signal = signal_data.get('signal', 'HOLD')
        if signal == 'BUY':
            self.buy_signals += 1
        elif signal == 'SELL':
            self.sell_signals += 1
        else:
            self.hold_signals += 1
        
        # FULL SIGNAL LOG
        if signal != 'HOLD':
            print(f"\n{'─'*80}")
            print(f"📡 {'BUY' if signal == 'BUY' else 'SELL'} SIGNAL DETECTED!")
            print(f"{'─'*80}")
            print(f"   Strategy:      {strategy}")
            print(f"   Version:       V{version}")
            print(f"   Token:         {token}")
            print(f"   ")
            print(f"   📈 MARKET DATA:")
            print(f"      Current Price:  ${signal_data.get('current_price', 0):,.4f}")
            print(f"      VWAP:           ${signal_data.get('vwap', 0):,.4f}")
            print(f"      Upper Band:     ${signal_data.get('upper_band', 0):,.4f}")
            print(f"      Lower Band:     ${signal_data.get('lower_band', 0):,.4f}")
            print(f"      ATR:            ${signal_data.get('atr', 0):,.4f}")
            print(f"      RSI:            {signal_data.get('rsi', 50):.1f}")
            print(f"      Momentum:       {signal_data.get('momentum', 0):.2f}%")
            print(f"   ")
            print(f"   🎯 TRADE SETUP:")
            print(f"      Entry:          ${signal_data.get('current_price', 0):,.4f}")
            print(f"      Target:         ${signal_data.get('target_price', 0):,.4f}")
            print(f"      Stop Loss:      ${signal_data.get('stop_loss', 0):,.4f}")
            print(f"   ")
            print(f"   💡 Reason: {signal_data.get('reason', 'N/A')}")
            print(f"{'─'*80}")
        
        # CSV log
        with open(self.signals_csv, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{self.cycle},{strategy},{token},{signal},"
                   f"{signal_data.get('current_price',0)},{signal_data.get('vwap',0)},"
                   f"{signal_data.get('upper_band',0)},{signal_data.get('lower_band',0)},"
                   f"{signal_data.get('rsi',0)},{signal_data.get('momentum',0)},"
                   f"\"{signal_data.get('reason','')}\"\n")
        
        self._write_log(f"SIGNAL | {strategy} V{version} | {token} | {signal}")
    
    def log_trade_open(self, position_id: str, strategy: str, version: int,
                       token: str, direction: str, price: float, size: float,
                       target: float, stop: float):
        """Log trade opened"""
        self.trades_opened += 1
        
        print(f"\n🎯 TRADE OPENED!")
        print(f"   ID:        {position_id}")
        print(f"   Strategy:  {strategy} V{version}")
        print(f"   Action:    {direction} {token}")
        print(f"   Entry:     ${price:,.4f}")
        print(f"   Size:      ${size:,.2f}")
        print(f"   Target:    ${target:,.4f}")
        print(f"   Stop:      ${stop:,.4f}")
        
        with open(self.trades_csv, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{self.cycle},{strategy},{version},{token},"
                   f"OPEN,{direction},{price},{size},{target},{stop},,,\n")
        
        self._write_log(f"TRADE OPEN | {position_id} | {direction} {token} @ ${price}")
    
    def log_trade_close(self, position_id: str, strategy: str, version: int,
                        token: str, direction: str, exit_price: float,
                        pnl: float, pnl_pct: float, reason: str):
        """Log trade closed"""
        self.trades_closed += 1
        
        if pnl >= 0:
            self.wins += 1
            emoji = "✅"
        else:
            self.losses += 1
            emoji = "❌"
        
        print(f"\n{emoji} TRADE CLOSED!")
        print(f"   ID:        {position_id}")
        print(f"   Strategy:  {strategy} V{version}")
        print(f"   Exit:      ${exit_price:,.4f}")
        print(f"   PnL:       ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   Reason:    {reason}")
        
        with open(self.trades_csv, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{self.cycle},{strategy},{version},{token},"
                   f"CLOSE,{direction},{exit_price},,,{pnl},{pnl_pct},{reason}\n")
        
        self._write_log(f"TRADE CLOSE | {position_id} | PnL: ${pnl:+.2f} | {reason}")
    
    def log_version_generated(self, strategy: str, old_version: int, new_version: int,
                              win_rate: float, improvements: List[str]):
        """Log new strategy version generated"""
        print(f"\n{'='*80}")
        print(f"🧬 NEW STRATEGY VERSION GENERATED!")
        print(f"{'='*80}")
        print(f"   Strategy:     {strategy}")
        print(f"   Old Version:  V{old_version}")
        print(f"   New Version:  V{new_version}")
        print(f"   Win Rate:     {win_rate:.1%}")
        print(f"   Improvements: {improvements}")
        print(f"{'='*80}")
        
        with open(self.versions_log, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"NEW VERSION: {strategy} V{old_version} → V{new_version}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Previous Win Rate: {win_rate:.1%}\n")
            f.write(f"Improvements: {improvements}\n")
            f.write(f"{'='*60}\n")
        
        self._write_log(f"VERSION | {strategy} V{old_version} → V{new_version}")
    
    def log_champion(self, strategy: str, version: int, win_rate: float, pnl: float):
        """Log strategy achieving CHAMPION status"""
        print(f"\n{'🏆'*30}")
        print(f"🏆 CHAMPION ACHIEVED! 🏆")
        print(f"{'🏆'*30}")
        print(f"   Strategy:  {strategy} V{version}")
        print(f"   Win Rate:  {win_rate:.1%}")
        print(f"   Total PnL: ${pnl:+,.2f}")
        print(f"   STATUS:    CHAMPION - TARGETS MET!")
        print(f"{'🏆'*30}\n")
        
        self._write_log(f"CHAMPION | {strategy} V{version} | WR: {win_rate:.1%} | PnL: ${pnl:+.2f}")
    
    def log_open_positions(self, positions: List, current_prices: Dict[str, float]):
        """
        Log all OPEN positions with FULL unrealized PnL details!
        Shows: Entry → Current price, % up/down, distance to target/stop, time in trade
        """
        open_pos = [p for p in positions if p.status == "OPEN"]
        
        if not open_pos:
            print(f"\n📊 NO OPEN POSITIONS")
            return
        
        print(f"\n{'='*100}")
        print(f"📊 OPEN POSITIONS - LIVE UNREALIZED PnL ({len(open_pos)} positions)")
        print(f"{'='*100}")
        
        total_unrealized = 0.0
        
        for pos in open_pos:
            current_price = current_prices.get(pos.token, pos.entry_price)
            
            # Calculate unrealized PnL
            if pos.direction == "BUY":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                dist_to_target = (pos.target - current_price) / current_price * 100
                dist_to_stop = (current_price - pos.stop) / current_price * 100
            else:  # SELL
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
                dist_to_target = (current_price - pos.target) / current_price * 100
                dist_to_stop = (pos.stop - current_price) / current_price * 100
            
            pnl_usd = pos.size * (pnl_pct / 100)
            total_unrealized += pnl_usd
            
            # Time in trade
            time_in_trade = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            # Emoji based on profit/loss
            if pnl_pct > 0:
                emoji = "🟢"
                status = "PROFIT"
            elif pnl_pct < 0:
                emoji = "🔴"
                status = "LOSS"
            else:
                emoji = "⚪"
                status = "BREAK-EVEN"
            
            print(f"\n   {emoji} {pos.strategy[:45]}")
            print(f"      {pos.direction} {pos.token}")
            print(f"      Entry:  ${pos.entry_price:,.4f}  →  Now: ${current_price:,.4f}")
            print(f"      PnL:    {status} {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            print(f"      🎯 Target: ${pos.target:,.4f} ({dist_to_target:+.2f}% away)")
            print(f"      🛑 Stop:   ${pos.stop:,.4f} ({dist_to_stop:.2f}% away)")
            print(f"      ⏱️  Time in trade: {time_in_trade:.0f} minutes")
        
        # Total unrealized PnL summary
        unrealized_emoji = "🟢" if total_unrealized >= 0 else "🔴"
        print(f"\n   {'─'*80}")
        print(f"   {unrealized_emoji} TOTAL UNREALIZED PnL: ${total_unrealized:+,.2f}")
        print(f"{'='*100}")
    
    def log_stats(self, capital: float, total_pnl: float, open_positions: int,
                  strategies: int, champions: int):
        """Log full statistics"""
        win_rate = self.wins / max(1, self.trades_closed) * 100
        
        print(f"\n{'='*100}")
        print(f"📊 TRADEPEX ADAPTIVE - FULL STATISTICS")
        print(f"{'='*100}")
        print(f"")
        print(f"💰 CAPITAL:")
        print(f"   Starting:      ${Config.STARTING_CAPITAL:,.2f}")
        print(f"   Current:       ${capital:,.2f}")
        print(f"   Total PnL:     ${total_pnl:+,.2f}")
        print(f"   Return:        {(total_pnl/Config.STARTING_CAPITAL)*100:+.2f}%")
        print(f"")
        print(f"📊 SIGNALS THIS SESSION:")
        print(f"   Total:         {self.total_signals}")
        print(f"   Buy Signals:   {self.buy_signals}")
        print(f"   Sell Signals:  {self.sell_signals}")
        print(f"   Hold:          {self.hold_signals}")
        print(f"")
        print(f"📈 TRADES:")
        print(f"   Opened:        {self.trades_opened}")
        print(f"   Closed:        {self.trades_closed}")
        print(f"   Open Now:      {open_positions}")
        print(f"   Wins:          {self.wins}")
        print(f"   Losses:        {self.losses}")
        print(f"   Win Rate:      {win_rate:.1f}%")
        print(f"")
        print(f"🧬 STRATEGIES:")
        print(f"   Active:        {strategies}")
        print(f"   Champions:     {champions}")
        print(f"   Target WR:     {Config.TARGET_WIN_RATE*100:.0f}%")
        print(f"{'='*100}\n")


# =============================================================================
# LLM STRATEGY OPTIMIZER - THE BRAIN THAT IMPROVES STRATEGIES!
# =============================================================================

class LLMStrategyOptimizer:
    """
    Uses LLM reasoning to analyze WHY strategies are losing and suggest improvements.
    This is the same approach as APEX RBI (Reasoning-Based Iteration).
    
    Instead of hardcoded rules, the LLM:
    1. Analyzes the trade history and loss patterns
    2. Reasons about WHY the strategy is failing
    3. Suggests specific parameter changes
    4. Can even recode the entire strategy if needed
    
    GOAL: Keep optimizing until 60%+ win rate achieved, then GO LIVE!
    """
    
    def __init__(self):
        self.optimization_history = []
        self.available_providers = []
        self._check_llm_availability()
    
    def _check_llm_availability(self):
        """Check which LLM providers are available"""
        if Config.DEEPSEEK_API_KEY:
            self.available_providers.append("deepseek")
            print("✅ DeepSeek API available for LLM reasoning")
        if Config.XAI_API_KEY:
            self.available_providers.append("xai")
            print("✅ XAI (Grok) API available for LLM reasoning")
        if Config.OPENAI_API_KEY and openai:
            self.available_providers.append("openai")
            print("✅ OpenAI API available for LLM reasoning")
        if Config.ANTHROPIC_API_KEY and anthropic:
            self.available_providers.append("anthropic")
            print("✅ Anthropic API available for LLM reasoning")
        
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
        """Call DeepSeek API"""
        if not openai:
            raise ImportError("openai package not installed")
        client = openai.OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=4000
        )
        return response.choices[0].message.content
    
    def _call_openai(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call OpenAI API"""
        if not openai:
            raise ImportError("openai package not installed")
        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=4000
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call Anthropic API"""
        if not anthropic:
            raise ImportError("anthropic package not installed")
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-sonnet-20240229", max_tokens=4000,
            system=system_prompt, messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_xai(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Call XAI (Grok) API"""
        if not openai:
            raise ImportError("openai package not installed")
        client = openai.OpenAI(api_key=Config.XAI_API_KEY, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=4000
        )
        return response.choices[0].message.content
    
    def analyze_and_optimize(self, strategy_id: str, performance: Dict, 
                             trade_history: List[Dict]) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Use LLM to analyze strategy performance and suggest improvements.
        
        Returns: (suggested_improvements, analysis_text) or (None, None) if failed
        """
        if not self.available_providers:
            print("⚠️  No LLM available, using fallback heuristics")
            return self._fallback_optimize(performance), "Fallback optimization"
        
        # Build the analysis prompt
        system_prompt = """You are an expert quantitative trading strategist.
Your task is to analyze why a trading strategy is underperforming and suggest SPECIFIC improvements.

You must respond in a STRICT JSON format:
{
    "analysis": "Your detailed analysis of why the strategy is failing",
    "root_causes": ["cause1", "cause2"],
    "improvements": {
        "entry_threshold": <suggested value 0.002-0.01>,
        "stop_loss_mult": <suggested value 1.0-3.0>,
        "take_profit_mult": <suggested value 1.0-3.0>,
        "rsi_oversold": <suggested value 20-40>,
        "rsi_overbought": <suggested value 60-80>
    },
    "reasoning": "Explain WHY each change will help",
    "confidence": <0.0 to 1.0>,
    "needs_full_recode": <true/false>
}

Be specific with numbers. Give exact values."""

        # Build trade summary
        trade_summary = self._build_trade_summary(trade_history)
        
        user_prompt = f"""Analyze this underperforming trading strategy:

## STRATEGY: {strategy_id}

## PERFORMANCE:
- Total Trades: {performance.get('trades', 0)}
- Win Rate: {performance.get('win_rate', 0):.1%} (TARGET: {Config.TARGET_WIN_RATE:.0%})
- Total PnL: ${performance.get('pnl', 0):.2f}
- Wins: {performance.get('wins', 0)}
- Losses: {performance.get('losses', 0)}

## RECENT TRADES:
{trade_summary}

## YOUR TASK:
1. Analyze WHY this strategy is losing money
2. Identify ROOT CAUSES
3. Suggest SPECIFIC parameter changes with EXACT numbers
4. Keep iterating until 60%+ win rate achieved!

Respond ONLY with valid JSON."""

        try:
            print(f"\n🤖 Sending to LLM for analysis...")
            response = self.call_llm(user_prompt, system_prompt)
            
            if not response:
                print("❌ LLM returned empty response")
                return self._fallback_optimize(performance), "Empty response"
            
            # Parse response
            improvements, analysis = self._parse_llm_response(response)
            
            if improvements:
                print(f"✅ LLM Analysis Complete:")
                print(f"   Analysis: {analysis[:200]}...")
                
                self.optimization_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'strategy_id': strategy_id,
                    'analysis': analysis,
                    'improvements': improvements
                })
                
                return improvements, analysis
            else:
                return self._fallback_optimize(performance), "Parse failed"
                
        except Exception as e:
            print(f"❌ LLM optimization failed: {e}")
            return self._fallback_optimize(performance), str(e)
    
    def _build_trade_summary(self, trades: List[Dict]) -> str:
        """Build trade summary for LLM"""
        if not trades:
            return "No trades yet"
        
        lines = []
        wins = len([t for t in trades if t.get('pnl', 0) > 0])
        losses = len([t for t in trades if t.get('pnl', 0) <= 0])
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        lines.append(f"Total: {len(trades)} trades, {wins} wins, {losses} losses, ${total_pnl:+.2f} PnL")
        
        # Last 10 trades
        for i, t in enumerate(trades[-10:], 1):
            outcome = "WIN ✅" if t.get('pnl', 0) > 0 else "LOSS ❌"
            lines.append(f"{i}. {t.get('token', '?')} {t.get('direction', '?')}: ${t.get('pnl', 0):+.2f} ({outcome}) - {t.get('reason', '')}")
        
        return "\n".join(lines)
    
    def _parse_llm_response(self, response: str) -> Tuple[Optional[Dict], str]:
        """Parse LLM JSON response"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                improvements = data.get('improvements', {})
                analysis = data.get('analysis', '')
                
                print(f"   Root causes: {data.get('root_causes', [])}")
                print(f"   Confidence: {data.get('confidence', 0):.0%}")
                
                return improvements, analysis
        except Exception as e:
            print(f"⚠️  JSON parse error: {e}")
        
        return None, ""
    
    def _fallback_optimize(self, performance: Dict) -> Dict:
        """Fallback heuristic optimization when no LLM available"""
        improvements = {}
        
        win_rate = performance.get('win_rate', 0)
        
        if win_rate < 0.4:
            # Very low win rate - tighten entries, widen stops
            improvements['entry_threshold'] = 0.003
            improvements['stop_loss_mult'] = 2.5
            improvements['rsi_oversold'] = 25
            improvements['rsi_overbought'] = 75
        elif win_rate < 0.5:
            # Below target - moderate adjustments
            improvements['entry_threshold'] = 0.004
            improvements['stop_loss_mult'] = 2.0
        
        print(f"📊 Fallback optimization: {improvements}")
        return improvements


# =============================================================================
# LEARNING ENGINE - TRACKS EVERYTHING FOR IMPROVEMENT
# =============================================================================

class LearningEngine:
    """
    Tracks all trading data for strategy improvement.
    
    Data is used to:
    1. Know which strategies are winning/losing
    2. Know which tokens work best
    3. Generate improved strategy versions
    """
    
    def __init__(self):
        self.data = {
            'strategies': defaultdict(lambda: {
                'version': 1,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'by_token': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0}),
                'is_champion': False,
                'champion_at': None
            }),
            'tokens': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0}),
            'total_trades': 0,
            'total_wins': 0,
            'total_pnl': 0,
            'iterations': 0,
            'champions': []
        }
        
        self._load()
    
    def _load(self):
        """Load previous learning data"""
        if Config.LEARNING_FILE.exists():
            try:
                with open(Config.LEARNING_FILE, 'r') as f:
                    saved = json.load(f)
                
                # Convert to defaultdicts
                if 'strategies' in saved:
                    for k, v in saved['strategies'].items():
                        self.data['strategies'][k] = v
                        if 'by_token' in v:
                            self.data['strategies'][k]['by_token'] = defaultdict(
                                lambda: {'trades': 0, 'wins': 0, 'pnl': 0},
                                v['by_token']
                            )
                
                for key in ['tokens', 'total_trades', 'total_wins', 'total_pnl', 'iterations', 'champions']:
                    if key in saved:
                        self.data[key] = saved[key]
                
                print(f"📚 Loaded learning data: {self.data['total_trades']} trades, {len(self.data['champions'])} champions")
            except Exception as e:
                print(f"⚠️ Could not load learning data: {e}")
    
    def save(self):
        """Save learning data"""
        try:
            # Convert defaultdicts to regular dicts
            save_data = {
                'strategies': {},
                'tokens': dict(self.data['tokens']),
                'total_trades': self.data['total_trades'],
                'total_wins': self.data['total_wins'],
                'total_pnl': self.data['total_pnl'],
                'iterations': self.data['iterations'],
                'champions': self.data['champions'],
                'last_saved': datetime.now().isoformat()
            }
            
            for k, v in self.data['strategies'].items():
                save_data['strategies'][k] = dict(v)
                if 'by_token' in v:
                    save_data['strategies'][k]['by_token'] = dict(v['by_token'])
            
            with open(Config.LEARNING_FILE, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save learning data: {e}")
    
    def record_trade(self, strategy: str, token: str, pnl: float, is_win: bool):
        """Record a completed trade"""
        # Strategy stats
        self.data['strategies'][strategy]['trades'] += 1
        self.data['strategies'][strategy]['total_pnl'] += pnl
        if is_win:
            self.data['strategies'][strategy]['wins'] += 1
        else:
            self.data['strategies'][strategy]['losses'] += 1
        
        # Strategy-token stats
        self.data['strategies'][strategy]['by_token'][token]['trades'] += 1
        self.data['strategies'][strategy]['by_token'][token]['pnl'] += pnl
        if is_win:
            self.data['strategies'][strategy]['by_token'][token]['wins'] += 1
        
        # Token stats
        self.data['tokens'][token]['trades'] += 1
        self.data['tokens'][token]['pnl'] += pnl
        if is_win:
            self.data['tokens'][token]['wins'] += 1
        
        # Totals
        self.data['total_trades'] += 1
        self.data['total_pnl'] += pnl
        if is_win:
            self.data['total_wins'] += 1
        
        # Save every 10 trades
        if self.data['total_trades'] % 10 == 0:
            self.save()
    
    def get_strategy_stats(self, strategy: str) -> Dict:
        """Get stats for a strategy"""
        stats = self.data['strategies'].get(strategy, {})
        trades = stats.get('trades', 0)
        
        return {
            'version': stats.get('version', 1),
            'trades': trades,
            'wins': stats.get('wins', 0),
            'win_rate': stats.get('wins', 0) / max(1, trades),
            'pnl': stats.get('total_pnl', 0),
            'is_champion': stats.get('is_champion', False),
            'best_tokens': self._get_best_tokens(strategy)
        }
    
    def _get_best_tokens(self, strategy: str) -> List[str]:
        """Get best performing tokens for a strategy"""
        by_token = self.data['strategies'].get(strategy, {}).get('by_token', {})
        
        good_tokens = []
        for token, stats in by_token.items():
            if stats['trades'] >= 3:
                win_rate = stats['wins'] / stats['trades']
                if win_rate >= 0.5:
                    good_tokens.append((token, win_rate))
        
        good_tokens.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in good_tokens[:5]]
    
    def should_iterate(self, strategy: str) -> bool:
        """Check if strategy needs new version"""
        stats = self.data['strategies'].get(strategy, {})
        
        # Already champion? No need
        if stats.get('is_champion', False):
            return False
        
        trades = stats.get('trades', 0)
        
        # Need minimum trades for evaluation
        if trades < Config.MIN_TRADES_FOR_EVALUATION:
            return False
        
        # Check if below target and enough trades for iteration
        win_rate = stats.get('wins', 0) / max(1, trades)
        
        # Iterate every N trades if not meeting target
        if trades % Config.ITERATION_INTERVAL == 0 and win_rate < Config.TARGET_WIN_RATE:
            return True
        
        return False
    
    def check_champion(self, strategy: str) -> bool:
        """Check if strategy achieved CHAMPION status"""
        stats = self.data['strategies'].get(strategy, {})
        
        if stats.get('is_champion', False):
            return True
        
        trades = stats.get('trades', 0)
        if trades < Config.MIN_TRADES_FOR_EVALUATION:
            return False
        
        win_rate = stats.get('wins', 0) / trades
        
        if win_rate >= Config.TARGET_WIN_RATE:
            # CHAMPION!
            self.data['strategies'][strategy]['is_champion'] = True
            self.data['strategies'][strategy]['champion_at'] = datetime.now().isoformat()
            self.data['champions'].append(strategy)
            self.save()
            return True
        
        return False
    
    def increment_version(self, strategy: str):
        """Increment strategy version"""
        self.data['strategies'][strategy]['version'] = \
            self.data['strategies'][strategy].get('version', 1) + 1
        self.data['iterations'] += 1
        self.save()


# =============================================================================
# STRATEGY VERSION GENERATOR - CREATES V1/V2/V3/V4 ETC
# =============================================================================

class StrategyVersionGenerator:
    """
    Generates new strategy versions (V1 → V2 → V3 → V4...) based on learning.
    
    Each version improves on the previous by:
    - Tightening entry conditions if too many losses
    - Widening stops if stopped out too much
    - Focusing on best-performing tokens
    - Adjusting indicator parameters
    """
    
    def __init__(self, learning: LearningEngine, logger: FullLogger):
        self.learning = learning
        self.logger = logger
        Config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    def generate_new_version(self, strategy_name: str, base_code: str = None) -> Optional[Path]:
        """
        Generate a new version of a strategy.
        
        Returns path to new strategy file.
        """
        stats = self.learning.get_strategy_stats(strategy_name)
        old_version = stats['version']
        new_version = old_version + 1
        win_rate = stats['win_rate']
        best_tokens = stats['best_tokens']
        
        # Determine improvements needed
        improvements = self._determine_improvements(stats)
        
        # Log it
        self.logger.log_version_generated(strategy_name, old_version, new_version, win_rate, improvements)
        
        # Generate code
        code = self._generate_code(strategy_name, new_version, improvements, best_tokens, win_rate)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{strategy_name}_V{new_version}.py"
        filepath = Config.GENERATED_DIR / filename
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        # Save meta
        meta = {
            'strategy_name': f"{strategy_name}_V{new_version}",
            'base_strategy': strategy_name,
            'version': new_version,
            'previous_win_rate': win_rate,
            'improvements': improvements,
            'best_tokens': best_tokens,
            'generated_at': datetime.now().isoformat()
        }
        
        meta_file = Config.GENERATED_DIR / f"{timestamp}_{strategy_name}_V{new_version}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Update learning data
        self.learning.increment_version(strategy_name)
        
        print(f"\n✅ Generated: {filepath.name}")
        
        return filepath
    
    def _determine_improvements(self, stats: Dict) -> List[str]:
        """Determine what improvements to make"""
        improvements = []
        
        win_rate = stats['win_rate']
        trades = stats['trades']
        
        if win_rate < 0.4:
            improvements.append('MUCH_TIGHTER_ENTRY')
            improvements.append('WIDER_STOPS')
        elif win_rate < 0.5:
            improvements.append('TIGHTER_ENTRY')
            improvements.append('ADJUST_STOPS')
        elif win_rate < 0.55:
            improvements.append('FINE_TUNE_ENTRY')
        
        if stats['pnl'] < 0:
            improvements.append('REDUCE_POSITION_SIZE')
            improvements.append('BETTER_EXIT_TIMING')
        
        if stats['best_tokens']:
            improvements.append('FOCUS_BEST_TOKENS')
        
        improvements.append('OPTIMIZE_PARAMETERS')
        
        return improvements
    
    def _generate_code(self, name: str, version: int, improvements: List[str],
                       best_tokens: List[str], prev_win_rate: float) -> str:
        """Generate the strategy Python code"""
        
        # Adjust parameters based on improvements
        if 'MUCH_TIGHTER_ENTRY' in improvements:
            entry_dev = 0.003
        elif 'TIGHTER_ENTRY' in improvements:
            entry_dev = 0.004
        else:
            entry_dev = 0.005
        
        if 'WIDER_STOPS' in improvements:
            stop_mult = 2.5
        elif 'ADJUST_STOPS' in improvements:
            stop_mult = 2.0
        else:
            stop_mult = 1.5
        
        position_pct = 0.08 if 'REDUCE_POSITION_SIZE' in improvements else 0.12
        rsi_oversold = 25 if 'MUCH_TIGHTER_ENTRY' in improvements else 30
        rsi_overbought = 75 if 'MUCH_TIGHTER_ENTRY' in improvements else 70
        
        # Build improvements string
        improvements_str = "\n".join(f"  - {imp}" for imp in improvements)
        best_tokens_repr = repr(best_tokens) if best_tokens else "[]"
        
        code = '''#!/usr/bin/env python3
"""
{name} - VERSION {version}
============================================================
Auto-generated by TradePexAdaptive
Generated: {timestamp}
Previous Win Rate: {prev_win_rate_str}
Improvements: {improvements}
Best Tokens: {best_tokens}

This version improves on V{prev_version} by applying:
{improvements_str}
"""

import numpy as np
import pandas as pd

class Strategy:
    """
    {name} Version {version}
    
    Improved strategy targeting {target_wr}% win rate.
    """
    
    def __init__(self):
        self.name = "{name}_V{version}"
        self.version = {version}
        self.best_tokens = {best_tokens_repr}
        
        # IMPROVED PARAMETERS
        self.entry_deviation = {entry_dev}  # Tightened entry threshold
        self.stop_atr_mult = {stop_mult}  # Adjusted stop loss
        self.target_atr_mult = 2.5
        self.position_pct = {position_pct}
        
        # RSI thresholds (tightened if needed)
        self.rsi_oversold = {rsi_oversold}
        self.rsi_overbought = {rsi_overbought}
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        ind = {{}}
        
        # VWAP
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        cum_vp = (typical * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum()
        ind['vwap'] = cum_vp / cum_vol
        
        # VWAP bands
        dev = np.abs(df['Close'] - ind['vwap'])
        std = dev.rolling(20).std()
        ind['upper'] = ind['vwap'] + 2 * std
        ind['lower'] = ind['vwap'] - 2 * std
        
        # ATR
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift()),
                np.abs(df['Low'] - df['Close'].shift())
            )
        )
        ind['atr'] = tr.rolling(14).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ind['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum
        ind['momentum'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
        
        return ind
    
    def generate_signal(self, df, current_price):
        """Generate trading signal with improved logic"""
        
        ind = self.calculate_indicators(df)
        
        vwap = float(ind['vwap'].iloc[-1])
        upper = float(ind['upper'].iloc[-1])
        lower = float(ind['lower'].iloc[-1])
        atr = float(ind['atr'].iloc[-1]) if not pd.isna(ind['atr'].iloc[-1]) else current_price * 0.02
        rsi = float(ind['rsi'].iloc[-1]) if not pd.isna(ind['rsi'].iloc[-1]) else 50
        momentum = float(ind['momentum'].iloc[-1]) if not pd.isna(ind['momentum'].iloc[-1]) else 0
        
        signal = "HOLD"
        reason = "No clear signal"
        target = 0
        stop = 0
        
        # Calculate deviations
        lower_dev = (lower - current_price) / current_price
        upper_dev = (current_price - upper) / current_price
        
        # IMPROVED BUY CONDITIONS
        if current_price < lower and lower_dev > self.entry_deviation:
            if rsi < self.rsi_oversold:  # Extra confirmation
                signal = "BUY"
                reason = f"V{self.version} BUY: Price ${{current_price:.2f}} < Lower ${{lower:.2f}}, RSI={{rsi:.0f}}"
                target = vwap
                stop = current_price - (atr * self.stop_atr_mult)
        
        # IMPROVED SELL CONDITIONS  
        elif current_price > upper and upper_dev > self.entry_deviation:
            if rsi > self.rsi_overbought:  # Extra confirmation
                signal = "SELL"
                reason = f"V{self.version} SELL: Price ${{current_price:.2f}} > Upper ${{upper:.2f}}, RSI={{rsi:.0f}}"
                target = vwap
                stop = current_price + (atr * self.stop_atr_mult)
        
        return {{
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'vwap': vwap,
            'upper_band': upper,
            'lower_band': lower,
            'target_price': target,
            'stop_loss': stop,
            'atr': atr,
            'rsi': rsi,
            'momentum': momentum
        }}


def get_strategy():
    return Strategy()
'''.format(
            name=name,
            version=version,
            timestamp=datetime.now().isoformat(),
            prev_win_rate_str=f"{prev_win_rate:.1%}",
            improvements=improvements,
            best_tokens=best_tokens,
            prev_version=version-1,
            improvements_str=improvements_str,
            target_wr=int(Config.TARGET_WIN_RATE*100),
            best_tokens_repr=best_tokens_repr,
            entry_dev=entry_dev,
            stop_mult=stop_mult,
            position_pct=position_pct,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )
        return code


# =============================================================================
# HTX CLIENT
# =============================================================================

class HTXClient:
    """Real-time market data from HTX"""
    
    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.cache_time = {}
        self.valid_tokens = set()
        self.invalid_tokens = set()
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        if symbol in self.invalid_tokens:
            return None
        
        # Cache for 3 seconds
        if symbol in self.cache and time.time() - self.cache_time.get(symbol, 0) < 3:
            return self.cache[symbol]
        
        try:
            resp = self.session.get(
                f"{Config.HTX_BASE_URL}/market/history/kline",
                params={"symbol": f"{symbol.lower()}usdt", "period": "1min", "size": 1},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok" and data.get("data"):
                    price = float(data["data"][0]["close"])
                    self.cache[symbol] = price
                    self.cache_time[symbol] = time.time()
                    self.valid_tokens.add(symbol)
                    return price
            
            self.invalid_tokens.add(symbol)
            return None
            
        except (requests.RequestException, ValueError, KeyError, Exception) as e:
            return None
    
    def get_candles(self, symbol: str, period: str = '15min', count: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV candles"""
        if symbol in self.invalid_tokens:
            return None
        
        try:
            resp = self.session.get(
                f"{Config.HTX_BASE_URL}/market/history/kline",
                params={"symbol": f"{symbol.lower()}usdt", "period": period, "size": count},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok" and data.get("data"):
                    df = pd.DataFrame(data["data"])
                    df = df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'amount': 'Volume'
                    })
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False).reset_index(drop=True)
                    self.valid_tokens.add(symbol)
                    return df
            
            return None
        except (requests.RequestException, ValueError, KeyError, Exception) as e:
            return None


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """Generates trading signals from market data"""
    
    def generate(self, df: pd.DataFrame, current_price: float, strategy_version: int = 1) -> Dict:
        """Generate signal using VWAP mean reversion"""
        
        # VWAP
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        cum_vp = (typical * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum()
        vwap = cum_vp / cum_vol
        
        # Bands
        dev = np.abs(df['Close'] - vwap)
        std = dev.rolling(20).std()
        upper = vwap + 2 * std
        lower = vwap - 2 * std
        
        # ATR
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift()))
        )
        atr = tr.rolling(14).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        # Momentum
        momentum = (df['Close'] / df['Close'].shift(10) - 1) * 100
        
        # Get latest values
        v = float(vwap.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        a = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        m = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0
        
        # Threshold based on version
        threshold = 0.005 - (strategy_version - 1) * 0.0005  # Tighter each version
        threshold = max(0.002, threshold)
        
        signal = "HOLD"
        reason = "Price within bands"
        target = 0
        stop = 0
        
        lower_dev = (l - current_price) / current_price
        upper_dev = (current_price - u) / current_price
        
        if current_price < l and lower_dev > threshold:
            signal = "BUY"
            reason = f"Price ${current_price:.2f} below lower ${l:.2f}"
            target = v
            stop = current_price - (a * 1.5)
        elif current_price > u and upper_dev > threshold:
            signal = "SELL"
            reason = f"Price ${current_price:.2f} above upper ${u:.2f}"
            target = v
            stop = current_price + (a * 1.5)
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'vwap': v,
            'upper_band': u,
            'lower_band': l,
            'target_price': target,
            'stop_loss': stop,
            'atr': a,
            'rsi': r,
            'momentum': m
        }


# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

class Position:
    def __init__(self, pid, strategy, version, token, direction, entry, size, target, stop):
        self.id = pid
        self.strategy = strategy
        self.version = version
        self.token = token
        self.direction = direction
        self.entry_price = entry
        self.size = size
        self.target = target
        self.stop = stop
        self.status = "OPEN"
        self.entry_time = datetime.now()  # Track when position opened


class PaperTrader:
    """Paper trading with position management"""
    
    def __init__(self, learning: LearningEngine, logger: FullLogger):
        self.learning = learning
        self.logger = logger
        self.capital = Config.STARTING_CAPITAL
        self.positions: Dict[str, Position] = {}
        self.closed: List[Position] = []
        self.total_pnl = 0
    
    def can_open(self, strategy: str, token: str) -> Tuple[bool, str]:
        """Check if can open position"""
        open_pos = [p for p in self.positions.values() if p.status == "OPEN"]
        
        if len(open_pos) >= Config.MAX_TOTAL_POSITIONS:
            return False, "Max positions"
        
        strat_pos = [p for p in open_pos if p.strategy == strategy]
        if len(strat_pos) >= Config.MAX_POSITIONS_PER_STRATEGY:
            return False, "Strategy limit"
        
        token_pos = [p for p in open_pos if p.token == token]
        if len(token_pos) >= Config.MAX_POSITIONS_PER_TOKEN:
            return False, "Token limit"
        
        # Already have this exact combo?
        if any(p.strategy == strategy and p.token == token for p in open_pos):
            return False, "Already open"
        
        return True, "OK"
    
    def open_position(self, strategy: str, version: int, token: str, signal: Dict) -> bool:
        """Open a position"""
        if signal['signal'] == 'HOLD':
            return False
        
        can, reason = self.can_open(strategy, token)
        if not can:
            return False
        
        size = self.capital * Config.MAX_POSITION_SIZE * Config.DEFAULT_LEVERAGE
        pid = f"{strategy}_{token}_{datetime.now().strftime('%H%M%S%f')}"
        
        pos = Position(
            pid, strategy, version, token,
            signal['signal'], signal['current_price'], size,
            signal['target_price'], signal['stop_loss']
        )
        
        self.positions[pid] = pos
        
        self.logger.log_trade_open(
            pid, strategy, version, token,
            signal['signal'], signal['current_price'], size,
            signal['target_price'], signal['stop_loss']
        )
        
        return True
    
    def check_exits(self, prices: Dict[str, float]):
        """Check and close positions"""
        for pid, pos in list(self.positions.items()):
            if pos.status != "OPEN":
                continue
            
            price = prices.get(pos.token)
            if not price:
                continue
            
            # Calculate PnL
            if pos.direction == "BUY":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price
            
            pnl = pos.size * pnl_pct
            
            # Check exits
            should_close = False
            reason = ""
            
            if pos.direction == "BUY":
                if price >= pos.target:
                    should_close, reason = True, "TARGET HIT"
                elif price <= pos.stop:
                    should_close, reason = True, "STOP LOSS"
            else:
                if price <= pos.target:
                    should_close, reason = True, "TARGET HIT"
                elif price >= pos.stop:
                    should_close, reason = True, "STOP LOSS"
            
            if should_close:
                self._close(pos, price, pnl, pnl_pct * 100, reason)
    
    def _close(self, pos: Position, exit_price: float, pnl: float, pnl_pct: float, reason: str):
        """Close position and record for LLM analysis"""
        pos.status = "CLOSED"
        self.capital += pnl
        self.total_pnl += pnl
        self.closed.append(pos)
        
        is_win = pnl >= 0
        
        # Record for learning
        self.learning.record_trade(pos.strategy, pos.token, pnl, is_win)
        
        # Log
        self.logger.log_trade_close(
            pos.id, pos.strategy, pos.version, pos.token,
            pos.direction, exit_price, pnl, pnl_pct, reason
        )
        
        # Store trade details for LLM analysis
        if not hasattr(self, 'trade_details'):
            self.trade_details = []
        
        self.trade_details.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': pos.strategy,
            'token': pos.token,
            'direction': pos.direction,
            'entry': pos.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'is_win': is_win
        })


# =============================================================================
# MAIN TRADEPEX ADAPTIVE ENGINE
# =============================================================================

class TradePexAdaptive:
    """
    THE ULTIMATE SELF-IMPROVING TRADING MACHINE!
    
    1. TRADES strategies
    2. LOGS everything (100% full logs)
    3. LEARNS from trades
    4. LLM ANALYZES why trades fail
    5. TradeAnalyzer identifies LOSS PATTERNS
    6. StrategyRecoder GENERATES improved .py files
    7. REPEATS until 60%+ win rate = CHAMPION = GO LIVE!
    """
    
    def __init__(self):
        self._banner()
        
        self.logger = FullLogger()
        self.learning = LearningEngine()
        self.llm_optimizer = LLMStrategyOptimizer()  # LLM for reasoning!
        self.trade_analyzer = TradeAnalyzer()  # Identifies loss patterns
        self.strategy_recoder = StrategyRecoder()  # Generates improved .py files
        self.htx = HTXClient()
        self.signal_gen = SignalGenerator()
        self.paper = PaperTrader(self.learning, self.logger)
        self.version_gen = StrategyVersionGenerator(self.learning, self.logger)
        
        # Track adaptive parameters per strategy
        self.strategy_params: Dict[str, AdaptiveParameters] = {}
        
        # Load strategies
        self.strategies = {}
        self._load_strategies()
        
        # Validate tokens
        self.valid_tokens = self._validate_tokens()
        
        self.cycle = 0
        self.running = True
    
    def _banner(self):
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                   ║
║   🚀 TRADEPEX ADAPTIVE - THE ULTIMATE SELF-IMPROVING TRADING MACHINE 🚀          ║
║                                                                                   ║
║   ✅ 100% FULL LOGGING - Every signal, every trade, every position!              ║
║   ✅ LLM REASONING - AI analyzes WHY trades fail and suggests fixes              ║
║   ✅ TRADE ANALYZER - Identifies loss patterns (tight stops, wrong trend, etc)   ║
║   ✅ STRATEGY RECODER - Generates improved .py files with optimized params       ║
║   ✅ GENERATES NEW VERSIONS - V1 → V2 → V3 → V4... until 60%+ win rate           ║
║   ✅ LIVE PnL DISPLAY - See every position's unrealized profit/loss              ║
║   ✅ ADAPTIVE PARAMETERS - Dynamic params per strategy based on performance      ║
║                                                                                   ║
║   🎯 Target: 60% Win Rate = CHAMPION STATUS = READY FOR LIVE TRADING!            ║
║                                                                                   ║
║   🌙 Trade → Learn → LLM Analyze → Recode Strategy → Repeat FOREVER 🌙          ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def _load_strategies(self):
        """Load all strategies from local and server directories"""
        # Check all possible directories
        dirs = [
            Config.STRATEGIES_DIR,           # Local: ./successful_strategies
            Config.SERVER_STRATEGIES_DIR,    # Server: /root/KEEP_SAFE/v1/APEX/successful_strategies
            Config.GENERATED_DIR             # Generated: ./generated_strategies
        ]
        
        for d in dirs:
            if d.exists():
                print(f"📂 Loading from: {d}")
                for f in d.glob("*.py"):
                    if '_meta' not in f.stem:
                        name = f.stem
                        version = 1
                        
                        # Extract version from name
                        v_match = re.search(r'_V(\d+)$', name)
                        if v_match:
                            version = int(v_match.group(1))
                        
                        self.strategies[name] = {
                            'file': f,
                            'version': version,
                            'name': name
                        }
                        print(f"✅ Loaded: {name} (V{version})")
        
        if not self.strategies:
            print("⚠️ No strategies found - creating defaults")
            self._create_defaults()
        
        print(f"\n🎯 TOTAL STRATEGIES: {len(self.strategies)}")
    
    def _create_defaults(self):
        """Create default strategies"""
        defaults = ['VWAP_Mean_Reversion', 'Momentum_Breakout', 'RSI_Reversal']
        
        for name in defaults:
            self.strategies[name] = {
                'file': None,
                'version': 1,
                'name': name
            }
            self.learning.data['strategies'][name]['version'] = 1
    
    def _validate_tokens(self) -> List[str]:
        """Validate tradeable tokens"""
        print(f"\n🔍 Validating tokens...")
        
        valid = []
        for token in Config.TRADEABLE_TOKENS:
            price = self.htx.get_price(token)
            if price:
                valid.append(token)
                print(f"   ✅ {token}: ${price:,.4f}")
            else:
                print(f"   ❌ {token}: Not available")
        
        print(f"\n💎 TRADING {len(valid)} TOKENS")
        return valid
    
    def run(self):
        """Main loop"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING TRADEPEX ADAPTIVE")
        print(f"{'='*80}")
        print(f"   Strategies: {len(self.strategies)}")
        print(f"   Tokens: {len(self.valid_tokens)}")
        print(f"   Check Interval: {Config.CHECK_INTERVAL}s")
        print(f"   Target Win Rate: {Config.TARGET_WIN_RATE*100:.0f}%")
        print(f"   Previous Trades: {self.learning.data['total_trades']}")
        print(f"   Champions: {len(self.learning.data['champions'])}")
        print(f"{'='*80}\n")
        
        while self.running:
            try:
                self.cycle += 1
                self._run_cycle()
                time.sleep(Config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 SHUTDOWN")
                self.learning.save()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                traceback.print_exc()
                time.sleep(30)
        
        print("\n📊 FINAL STATS:")
        self.logger.log_stats(
            self.paper.capital, self.paper.total_pnl,
            len([p for p in self.paper.positions.values() if p.status == "OPEN"]),
            len(self.strategies), len(self.learning.data['champions'])
        )
    
    def _run_cycle(self):
        """Run one trading cycle"""
        # Get prices
        prices = {}
        for token in self.valid_tokens:
            p = self.htx.get_price(token)
            if p:
                prices[token] = p
        
        self.logger.log_cycle_start(self.cycle, prices)
        
        # Check exits
        self.paper.check_exits(prices)
        
        # Check for strategy improvements
        self._check_improvements()
        
        # Generate signals and trade
        for strat_name, strat_info in self.strategies.items():
            version = self.learning.data['strategies'][strat_name].get('version', 1)
            
            for token in self.valid_tokens:
                self._process(strat_name, version, token, prices)
        
        # Show OPEN POSITIONS with UNREALIZED PnL (the key feature from tradeadapt!)
        self.logger.log_open_positions(
            list(self.paper.positions.values()),
            prices
        )
        
        # Show stats
        self.logger.log_stats(
            self.paper.capital, self.paper.total_pnl,
            len([p for p in self.paper.positions.values() if p.status == "OPEN"]),
            len(self.strategies), len(self.learning.data['champions'])
        )
    
    def _process(self, strategy: str, version: int, token: str, prices: Dict):
        """Process a strategy-token combination"""
        price = prices.get(token)
        if not price:
            return
        
        # Check if can open first
        can, _ = self.paper.can_open(strategy, token)
        if not can:
            return
        
        # Get candles
        df = self.htx.get_candles(token, '15min', 100)
        if df is None or len(df) < 50:
            return
        
        # Generate signal
        signal = self.signal_gen.generate(df, price, version)
        
        # Log signal
        self.logger.log_signal(strategy, version, token, signal)
        
        # Open position if signal
        if signal['signal'] != 'HOLD':
            self.paper.open_position(strategy, version, token, signal)
    
    def _check_improvements(self):
        """
        Check if any strategy needs improvement using LLM reasoning + TradeAnalyzer.
        
        THE KEY LOOP:
        1. Check if strategy is already CHAMPION (60%+ win rate)
        2. TradeAnalyzer identifies LOSS PATTERNS
        3. LLM analyzes WHY it's failing
        4. StrategyRecoder generates improved .py file
        5. Trade the new version
        6. Repeat until CHAMPION status = READY TO GO LIVE!
        """
        for strat_name in list(self.strategies.keys()):
            # Check champion status first - if 60%+ win rate, GO LIVE!
            if self.learning.check_champion(strat_name):
                stats = self.learning.get_strategy_stats(strat_name)
                self.logger.log_champion(
                    strat_name, stats['version'],
                    stats['win_rate'], stats['pnl']
                )
                print(f"\n🏆 CHAMPION ACHIEVED! {strat_name} is READY TO GO LIVE!")
                continue
            
            # Get strategy stats
            stats = self.learning.get_strategy_stats(strat_name)
            
            # Run optimization every OPTIMIZATION_INTERVAL cycles
            if self.cycle % Config.OPTIMIZATION_INTERVAL == 0 and stats['trades'] >= Config.MIN_TRADES_FOR_ANALYSIS:
                print(f"\n{'='*100}")
                print(f"🤖 OPTIMIZATION ANALYSIS: {strat_name[:50]}")
                print(f"{'='*100}")
                print(f"   Current Win Rate: {stats['win_rate']:.1%} (TARGET: {Config.TARGET_WIN_RATE:.0%})")
                print(f"   Total Trades: {stats['trades']}")
                print(f"   PnL: ${stats['pnl']:+.2f}")
                
                # 1. TradeAnalyzer: Identify loss patterns
                print(f"\n🔍 TRADE ANALYZER - Loss Pattern Detection:")
                suggestions = self.trade_analyzer.get_optimization_suggestions(strat_name)
                patterns = suggestions.get('patterns', [])
                if patterns:
                    print(f"   Patterns Found: {patterns}")
                    for suggestion in suggestions.get('suggestions', []):
                        print(f"   💡 {suggestion}")
                else:
                    print(f"   No clear loss patterns identified yet")
                
                # 2. LLM: Deep analysis with reasoning
                trade_details = getattr(self.paper, 'trade_details', [])
                strategy_trades = [t for t in trade_details if t.get('strategy') == strat_name][-20:]
                
                print(f"\n🤖 LLM REASONING:")
                improvements, analysis = self.llm_optimizer.analyze_and_optimize(
                    strat_name, stats, strategy_trades
                )
                
                if improvements:
                    print(f"\n📝 LLM SUGGESTED IMPROVEMENTS:")
                    for key, value in improvements.items():
                        print(f"   {key}: {value}")
                    
                    # 3. Create/Update AdaptiveParameters
                    params = self.strategy_params.get(strat_name, AdaptiveParameters())
                    params.version += 1
                    params.optimization_count += 1
                    params.last_optimized = datetime.now().isoformat()
                    
                    # Apply LLM improvements to params
                    if 'entry_threshold' in improvements:
                        params.min_deviation_percent = float(improvements['entry_threshold'])
                    if 'stop_loss_mult' in improvements:
                        params.stop_loss_atr_multiplier = float(improvements['stop_loss_mult'])
                    if 'take_profit_mult' in improvements:
                        params.take_profit_atr_multiplier = float(improvements['take_profit_mult'])
                    
                    self.strategy_params[strat_name] = params
                    
                    # 4. StrategyRecoder: Generate improved .py file
                    perf = self.trade_analyzer.strategy_performance.get(
                        strat_name, 
                        StrategyPerformance(strategy_id=strat_name)
                    )
                    perf.total_trades = stats['trades']
                    perf.win_rate = stats['win_rate']
                    perf.total_pnl = stats['pnl']
                    
                    strat_info = self.strategies.get(strat_name, {})
                    original_file = strat_info.get('file', Path(f"./{strat_name}.py"))
                    
                    new_file = self.strategy_recoder.recode_strategy(
                        strategy_id=strat_name,
                        params=params,
                        performance=perf,
                        original_file=original_file
                    )
                    
                    if new_file:
                        new_name = new_file.stem
                        self.strategies[new_name] = {
                            'file': new_file,
                            'version': params.version,
                            'name': new_name
                        }
                        
                        print(f"\n✅ STRATEGY RECODED: {new_file.name}")
                        print(f"   Saved improved .py file with V{params.version} parameters!")
                    else:
                        # Fallback to version generator
                        new_file = self.version_gen.generate_new_version(strat_name)
                        if new_file:
                            new_name = new_file.stem
                            self.strategies[new_name] = {
                                'file': new_file,
                                'version': params.version,
                                'name': new_name
                            }
                
                # Print full analysis report
                self.trade_analyzer.print_analysis(strat_name)
            
            # Also check if needs iteration based on trade count
            elif self.learning.should_iterate(strat_name):
                new_file = self.version_gen.generate_new_version(strat_name)
                
                if new_file:
                    new_name = new_file.stem
                    new_version = self.learning.data['strategies'][strat_name]['version']
                    
                    self.strategies[new_name] = {
                        'file': new_file,
                        'version': new_version,
                        'name': new_name
                    }
    
    def record_trade_context(self, pos, exit_price: float, pnl: float, pnl_pct: float, reason: str, context: Dict = None):
        """Record a closed trade with full context for TradeAnalyzer"""
        trade = TradeContext(
            timestamp=datetime.now().isoformat(),
            position_id=pos.id,
            strategy_id=pos.strategy,
            symbol=pos.token,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            target_price=pos.target,
            stop_loss=pos.stop,
            size_usd=pos.size,
            pnl_usd=pnl,
            pnl_percent=pnl_pct,
            status="CLOSED",
            exit_reason=reason,
            vwap=context.get('vwap', 0) if context else 0,
            deviation_percent=context.get('deviation', 0) if context else 0,
            volatility_24h=context.get('volatility', 0) if context else 0,
            trend_direction=context.get('trend', '') if context else '',
            holding_time_minutes=(datetime.now() - pos.entry_time).total_seconds() / 60
        )
        self.trade_analyzer.add_trade(trade)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    engine = TradePexAdaptive()
    engine.run()
