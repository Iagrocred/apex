#!/usr/bin/env python3
"""
TRADEPEX ADAPTIVE - Dynamic Live Strategy Analyzer & Optimizer
=================================================================
This system analyzes live paper trading performance, identifies why strategies
are losing, and dynamically adjusts parameters until profitability is achieved.

Key Features:
1. Detailed Trade Logging - Captures full market context for each trade
2. Loss Pattern Analysis - Identifies why trades fail
3. Dynamic Parameter Adjustment - Auto-adjusts strategy parameters based on analysis
4. Adaptive Optimization Loop - Continues until target profitability is reached
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

# =============================================================================
# CONFIGURATION - ADAPTIVE TRADING
# =============================================================================

class Config:
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.15  # 15% per trade
    DEFAULT_LEVERAGE = 3
    HTX_BASE_URL = "https://api.huobi.pro"
    TRADEABLE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX']
    STRATEGIES_DIR = Path("./successfule_strategies")  # Local path
    CHECK_INTERVAL = 30
    
    # Position Limits
    MAX_TOTAL_POSITIONS = 8
    MAX_POSITIONS_PER_STRATEGY = 2
    MAX_POSITIONS_PER_TOKEN = 2
    
    # Adaptive Optimization Settings
    MIN_TRADES_FOR_ANALYSIS = 5       # Minimum trades before analyzing
    TARGET_WIN_RATE = 0.55            # Target 55% win rate
    TARGET_PROFIT_FACTOR = 1.3        # Target profit factor
    OPTIMIZATION_INTERVAL = 10        # Cycles between parameter adjustments
    MAX_CONSECUTIVE_LOSSES = 3        # Pause strategy after this many losses
    
    # Log Files
    TRADE_LOG_FILE = Path("./tradepex_trades.json")
    ANALYSIS_LOG_FILE = Path("./tradepex_analysis.json")
    PARAMETER_LOG_FILE = Path("./tradepex_parameters.json")

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
    
    # Duration
    holding_time_minutes: float = 0.0
    
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
    # Entry Conditions
    min_deviation_percent: float = 0.3   # Minimum deviation from band to enter
    max_volatility_percent: float = 5.0  # Max 24h volatility to enter
    min_volume_ratio: float = 0.5        # Min volume vs average
    
    # Risk Management
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    max_holding_periods: int = 20        # Max periods before forced exit
    
    # Band Calculation
    std_dev_multiplier: float = 2.0      # For VWAP bands
    lookback_period: int = 20            # For rolling calculations
    
    # Position Sizing
    position_size_percent: float = 0.15  # % of capital per trade
    
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
    
    def to_dict(self) -> dict:
        return asdict(self)

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
# ADAPTIVE STRATEGY EXECUTOR
# =============================================================================

class AdaptiveStrategyExecutor:
    """Strategy executor with adaptive parameters"""
    
    def __init__(self, params: AdaptiveParameters):
        self.params = params
    
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
        """Generate trading signal with full context"""
        context = self.calculate_market_context(df)
        
        signal = "HOLD"
        reason = "No signal"
        target_price = 0.0
        stop_loss = 0.0
        
        # Calculate deviations
        lower_deviation = 0.0
        upper_deviation = 0.0
        
        if context['lower_band'] > 0:
            lower_deviation = (context['lower_band'] - current_price) / current_price * 100
        if context['upper_band'] > 0:
            upper_deviation = (current_price - context['upper_band']) / current_price * 100

        # Filter Conditions
        volatility_ok = context['volatility_24h'] < self.params.max_volatility_percent
        volume_ok = context['volume_ratio'] >= self.params.min_volume_ratio
        
        # BUY Signal - Price below lower band
        if (current_price < context['lower_band'] and 
            lower_deviation > self.params.min_deviation_percent and
            volatility_ok and volume_ok):
            
            signal = "BUY"
            reason = f"Price ${current_price:.2f} below lower band ${context['lower_band']:.2f} (dev: {lower_deviation:.2f}%)"
            target_price = current_price + (context['atr'] * self.params.take_profit_atr_multiplier)
            stop_loss = current_price - (context['atr'] * self.params.stop_loss_atr_multiplier)

        # SELL Signal - Price above upper band
        elif (current_price > context['upper_band'] and 
              upper_deviation > self.params.min_deviation_percent and
              volatility_ok and volume_ok):
            
            signal = "SELL"
            reason = f"Price ${current_price:.2f} above upper band ${context['upper_band']:.2f} (dev: {upper_deviation:.2f}%)"
            target_price = current_price - (context['atr'] * self.params.take_profit_atr_multiplier)
            stop_loss = current_price + (context['atr'] * self.params.stop_loss_atr_multiplier)
        
        else:
            if not volatility_ok:
                reason = f"Volatility {context['volatility_24h']:.2f}% > max {self.params.max_volatility_percent}%"
            elif not volume_ok:
                reason = f"Volume ratio {context['volume_ratio']:.2f} < min {self.params.min_volume_ratio}"
            elif lower_deviation > 0 and lower_deviation < self.params.min_deviation_percent:
                reason = f"Deviation {lower_deviation:.2f}% < min {self.params.min_deviation_percent}%"
            elif upper_deviation > 0 and upper_deviation < self.params.min_deviation_percent:
                reason = f"Deviation {upper_deviation:.2f}% < min {self.params.min_deviation_percent}%"
            else:
                reason = "Price within bands"

        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            **context,
            'deviation_percent': max(lower_deviation, upper_deviation)
        }

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

class AdaptivePaperTradingEngine:
    """Paper trading engine with trade analysis and adaptive learning"""
    
    def __init__(self):
        self.positions: Dict[str, AdaptivePaperTrade] = {}
        self.capital = Config.STARTING_CAPITAL
        self.trade_history: List[TradeContext] = []
        self.analyzer = TradeAnalyzer()
        
        # Strategy-specific adaptive parameters
        self.strategy_params: Dict[str, AdaptiveParameters] = {}
    
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
        """Open position with full context"""
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
            trend_direction=signal.get('trend', '')
        )
        
        self.positions[position_id] = position
        
        print(f"🎯 OPENED: {position_id[:50]}")
        print(f"   {signal['signal']} {symbol} @ ${signal['current_price']:.2f}")
        print(f"   Target: ${signal.get('target_price', 0):.2f} | Stop: ${signal.get('stop_loss', 0):.2f}")
        print(f"   Deviation: {signal.get('deviation_percent', 0):.2f}% | Volatility: {signal.get('volatility_24h', 0):.2f}%")
        
        return position_id
    
    def check_exits(self, current_prices: Dict):
        """Check and execute exits"""
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
        """Analyze and optimize strategy parameters"""
        if strategy_id not in self.analyzer.strategy_performance:
            return
        
        perf = self.analyzer.strategy_performance[strategy_id]
        
        if perf.total_trades < Config.MIN_TRADES_FOR_ANALYSIS:
            return
        
        # Check if optimization needed
        needs_optimization = (
            perf.win_rate < Config.TARGET_WIN_RATE or
            perf.profit_factor < Config.TARGET_PROFIT_FACTOR
        )
        
        if not needs_optimization:
            return
        
        print(f"\n🔧 OPTIMIZING: {strategy_id[:50]}")
        print(f"   Current Win Rate: {perf.win_rate:.1%} (target: {Config.TARGET_WIN_RATE:.0%})")
        print(f"   Current Profit Factor: {perf.profit_factor:.2f} (target: {Config.TARGET_PROFIT_FACTOR})")
        
        # Get patterns and adjust
        patterns = self.analyzer.identify_loss_patterns(strategy_id)
        params = self.get_params(strategy_id)
        
        for pattern in patterns:
            params.adjust_for_losses(pattern)
        
        # Reset consecutive losses to unpause strategy
        if perf.is_paused and len(patterns) > 0:
            perf.consecutive_losses = 0
            perf.is_paused = False
            print(f"   ✅ Strategy UNPAUSED after parameter adjustment")
        
        print(f"   📊 New Parameters: min_dev={params.min_deviation_percent:.2f}%, "
              f"max_vol={params.max_volatility_percent:.2f}%, "
              f"sl_mult={params.stop_loss_atr_multiplier:.2f}")
    
    def save_state(self):
        """Save trade history and parameters to files"""
        # Save trades
        trades_data = [t.to_dict() for t in self.trade_history]
        with open(Config.TRADE_LOG_FILE, 'w') as f:
            json.dump(trades_data, f, indent=2, default=str)
        
        # Save parameters
        params_data = {k: v.to_dict() for k, v in self.strategy_params.items()}
        with open(Config.PARAMETER_LOG_FILE, 'w') as f:
            json.dump(params_data, f, indent=2)

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
        
        # Create strategy executors with adaptive parameters
        self.executors: Dict[str, AdaptiveStrategyExecutor] = {}
        for strategy_id in self.strategies:
            params = self.paper_engine.get_params(strategy_id)
            self.executors[strategy_id] = AdaptiveStrategyExecutor(params)
    
    def load_strategies(self) -> Dict:
        """Load strategies from directory"""
        strategies = {}
        
        print(f"🔍 Looking for strategies in: {Config.STRATEGIES_DIR}")
        
        if Config.STRATEGIES_DIR.exists():
            py_files = [f for f in Config.STRATEGIES_DIR.glob("*.py") if '_meta' not in str(f)]
            print(f"📁 Found {len(py_files)} strategy files")
            
            for py_file in py_files:
                strategy_id = py_file.stem
                strategies[strategy_id] = {'py_file': py_file}
                print(f"✅ LOADED: {strategy_id}")
        else:
            print(f"⚠️  Strategies directory not found: {Config.STRATEGIES_DIR}")
        
        print(f"🎯 Total strategies: {len(strategies)}")
        return strategies
    
    def execute_strategy_for_token(self, strategy_id: str, token: str):
        """Execute strategy with adaptive parameters"""
        try:
            df = self.htx_client.fetch_candles(token, '15min', 100)
            if df is None or len(df) < 50:
                return
            
            current_price = self.htx_client.get_current_price(token)
            if not current_price:
                return
            
            # Get executor with current adaptive parameters
            executor = self.executors[strategy_id]
            signal = executor.generate_signal(df, current_price)
            
            if signal['signal'] != 'HOLD':
                print(f"\n🚀 SIGNAL: {strategy_id[:40]} - {signal['signal']} {token}")
                print(f"   Reason: {signal['reason']}")
                self.paper_engine.open_position(strategy_id, token, signal)
            
        except Exception as e:
            print(f"❌ Error: {strategy_id[:30]} {token}: {e}")
    
    def display_status(self):
        """Display current status"""
        open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
        closed_positions = [p for p in self.paper_engine.positions.values() if p.status == "CLOSED"]
        total_pnl = sum(p.pnl for p in closed_positions)
        
        print(f"\n{'='*80}")
        print(f"🔄 TRADEPEX ADAPTIVE - Dynamic Strategy Optimizer")
        print(f"{'='*80}")
        print(f"⏰ Cycle: {self.cycle_count} | Runtime: {datetime.now() - self.start_time}")
        print(f"💰 Capital: ${self.paper_engine.capital:.2f} | Total PnL: ${total_pnl:+.2f}")
        print(f"📊 Open: {len(open_positions)}/{Config.MAX_TOTAL_POSITIONS} | Closed: {len(closed_positions)}")
        
        # Performance summary
        if closed_positions:
            wins = len([p for p in closed_positions if p.pnl > 0])
            win_rate = wins / len(closed_positions) * 100
            print(f"📈 Win Rate: {win_rate:.1f}% ({wins}/{len(closed_positions)})")
        
        if open_positions:
            print(f"\n📊 OPEN POSITIONS:")
            for position in open_positions:
                current_price = self.htx_client.get_current_price(position.symbol) or position.entry_price
                if position.direction == "BUY":
                    pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    pnl_percent = (position.entry_price - current_price) / position.entry_price * 100
                
                print(f"   {position.strategy_id[:25]:<25} {position.symbol:<5} {position.direction:<4} "
                      f"Entry: ${position.entry_price:<9.2f} PnL: {pnl_percent:+.2f}%")
        
        print(f"{'='*80}\n")
    
    def run_adaptive(self):
        """Run with adaptive learning"""
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
                
                # Look for new opportunities
                open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
                if len(open_positions) < Config.MAX_TOTAL_POSITIONS:
                    for strategy_id in self.strategies:
                        for token in Config.TRADEABLE_TOKENS:
                            self.execute_strategy_for_token(strategy_id, token)
                            time.sleep(0.3)
                else:
                    print(f"⏸️  Position limit reached ({len(open_positions)}/{Config.MAX_TOTAL_POSITIONS})")
                
                # Periodic optimization
                if self.cycle_count % Config.OPTIMIZATION_INTERVAL == 0:
                    print("\n🔧 RUNNING OPTIMIZATION CYCLE...")
                    for strategy_id in self.strategies:
                        self.paper_engine.optimize_strategy(strategy_id)
                    
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
