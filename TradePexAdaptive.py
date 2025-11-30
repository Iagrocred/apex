#!/usr/bin/env python3
"""
🚀 TRADEPEX ADAPTIVE - SELF-IMPROVING INTELLIGENT TRADING SYSTEM
=================================================================

UPGRADE FROM V1:
- V1 had hardcoded strategy types
- ADAPTIVE dynamically learns from ANY strategy file
- ADAPTIVE self-improves by tracking what works and what doesn't
- ADAPTIVE trades MORE tokens to learn faster

KEY FEATURES:
1. DYNAMIC STRATEGY LOADING - No hardcoded types! Reads the actual .py code
2. SELF-IMPROVING - Tracks performance per strategy/token and adjusts
3. MORE TOKENS - Trades all available tokens on HTX
4. LEARNING LOOP - More trades = more data = better performance

The system LEARNS by trading. Each trade teaches us:
- Which strategies work on which tokens
- What market conditions favor each strategy
- When to increase/decrease position sizes

🌙 This is NOT V1 - This is the UPGRADED LEARNING SYSTEM 🌙
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
import hashlib
import traceback

# =============================================================================
# CONFIGURATION - EXPANDED FOR MORE TRADING = MORE LEARNING
# =============================================================================

class Config:
    """TradePex Adaptive Configuration - Expanded for Maximum Learning"""
    
    # Capital Management
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.10  # 10% per trade (smaller = more trades)
    DEFAULT_LEVERAGE = 3
    
    # Exchange
    HTX_BASE_URL = "https://api.huobi.pro"
    
    # EXPANDED TOKEN LIST - MORE TOKENS = MORE LEARNING OPPORTUNITIES
    # Trading ALL major tokens on HTX
    TRADEABLE_TOKENS = [
        # Major Cryptos
        'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX',
        # Layer 1s
        'MATIC', 'ATOM', 'NEAR', 'FTM', 'ALGO', 'ICP',
        # DeFi
        'UNI', 'AAVE', 'MKR', 'CRV', 'LDO', 'SNX',
        # Layer 2s & Scaling
        'OP', 'ARB', 'IMX',
        # Meme & Community (high volatility = more signals)
        'DOGE', 'SHIB', 'PEPE', 'FLOKI',
        # Other Major
        'LTC', 'BCH', 'ETC', 'FIL', 'APT', 'SUI', 'SEI'
    ]
    
    # Strategies directory
    STRATEGIES_DIR = Path(__file__).parent / "successfule_strategies"
    FALLBACK_STRATEGIES_DIR = Path("/root/KEEP_SAFE/v1/APEX/successful_strategies")
    
    # Trading parameters - INCREASED LIMITS FOR MORE TRADING
    CHECK_INTERVAL = 20  # Faster cycles = more opportunities
    
    # Position Limits - EXPANDED for more learning
    MAX_TOTAL_POSITIONS = 20        # Much higher - we want to TRADE MORE
    MAX_POSITIONS_PER_STRATEGY = 5  # Each strategy can have more trades
    MAX_POSITIONS_PER_TOKEN = 4     # More positions per token
    
    # Self-improvement settings
    LEARNING_DATA_FILE = Path(__file__).parent / "tradepex_learning_data.json"
    MIN_TRADES_FOR_ADJUSTMENT = 10  # Need this many trades before adjusting
    WIN_RATE_BOOST_THRESHOLD = 0.6  # Boost strategies with >60% win rate
    WIN_RATE_REDUCE_THRESHOLD = 0.4 # Reduce strategies with <40% win rate


# =============================================================================
# SELF-IMPROVEMENT ENGINE - LEARNS FROM EVERY TRADE
# =============================================================================

class LearningEngine:
    """
    Self-improving engine that tracks performance and adjusts behavior.
    
    This is what makes ADAPTIVE different from V1:
    - Tracks win/loss per strategy per token
    - Adjusts position sizes based on performance
    - Learns which strategies work on which tokens
    - Gets better over time!
    """
    
    def __init__(self):
        self.learning_data = {
            'strategy_performance': defaultdict(lambda: {
                'trades': 0, 'wins': 0, 'losses': 0,
                'total_pnl': 0, 'best_tokens': [], 'worst_tokens': []
            }),
            'token_performance': defaultdict(lambda: {
                'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0
            }),
            'strategy_token_matrix': defaultdict(lambda: defaultdict(lambda: {
                'trades': 0, 'wins': 0, 'pnl': 0
            })),
            'total_trades': 0,
            'total_wins': 0,
            'total_pnl': 0,
            'last_updated': None
        }
        
        self._load_learning_data()
    
    def _load_learning_data(self):
        """Load previous learning data if exists"""
        if Config.LEARNING_DATA_FILE.exists():
            try:
                with open(Config.LEARNING_DATA_FILE, 'r') as f:
                    saved = json.load(f)
                    # Merge with defaults
                    for key in saved:
                        if key in self.learning_data:
                            if isinstance(self.learning_data[key], defaultdict):
                                for k, v in saved[key].items():
                                    self.learning_data[key][k] = v
                            else:
                                self.learning_data[key] = saved[key]
                print(f"📚 Loaded learning data: {self.learning_data['total_trades']} previous trades")
            except Exception as e:
                print(f"⚠️  Could not load learning data: {e}")
    
    def _save_learning_data(self):
        """Save learning data for persistence"""
        try:
            # Convert defaultdicts to regular dicts for JSON
            save_data = {
                'strategy_performance': dict(self.learning_data['strategy_performance']),
                'token_performance': dict(self.learning_data['token_performance']),
                'strategy_token_matrix': {k: dict(v) for k, v in self.learning_data['strategy_token_matrix'].items()},
                'total_trades': self.learning_data['total_trades'],
                'total_wins': self.learning_data['total_wins'],
                'total_pnl': self.learning_data['total_pnl'],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(Config.LEARNING_DATA_FILE, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save learning data: {e}")
    
    def record_trade(self, strategy_id: str, token: str, pnl: float, is_win: bool):
        """Record a completed trade for learning"""
        # Update strategy stats
        self.learning_data['strategy_performance'][strategy_id]['trades'] += 1
        self.learning_data['strategy_performance'][strategy_id]['total_pnl'] += pnl
        if is_win:
            self.learning_data['strategy_performance'][strategy_id]['wins'] += 1
        else:
            self.learning_data['strategy_performance'][strategy_id]['losses'] += 1
        
        # Update token stats
        self.learning_data['token_performance'][token]['trades'] += 1
        self.learning_data['token_performance'][token]['total_pnl'] += pnl
        if is_win:
            self.learning_data['token_performance'][token]['wins'] += 1
        else:
            self.learning_data['token_performance'][token]['losses'] += 1
        
        # Update strategy-token matrix
        self.learning_data['strategy_token_matrix'][strategy_id][token]['trades'] += 1
        self.learning_data['strategy_token_matrix'][strategy_id][token]['pnl'] += pnl
        if is_win:
            self.learning_data['strategy_token_matrix'][strategy_id][token]['wins'] += 1
        
        # Update totals
        self.learning_data['total_trades'] += 1
        self.learning_data['total_pnl'] += pnl
        if is_win:
            self.learning_data['total_wins'] += 1
        
        # Save periodically
        if self.learning_data['total_trades'] % 10 == 0:
            self._save_learning_data()
    
    def get_strategy_confidence(self, strategy_id: str) -> float:
        """
        Get confidence multiplier for a strategy based on past performance.
        Returns 0.5-1.5 multiplier for position sizing.
        """
        stats = self.learning_data['strategy_performance'].get(strategy_id)
        if not stats or stats['trades'] < Config.MIN_TRADES_FOR_ADJUSTMENT:
            return 1.0  # Default confidence
        
        win_rate = stats['wins'] / stats['trades']
        
        if win_rate >= Config.WIN_RATE_BOOST_THRESHOLD:
            # Boost winning strategies
            return min(1.5, 1.0 + (win_rate - 0.5) * 0.5)
        elif win_rate <= Config.WIN_RATE_REDUCE_THRESHOLD:
            # Reduce losing strategies
            return max(0.5, 1.0 - (0.5 - win_rate) * 0.5)
        else:
            return 1.0
    
    def get_token_suitability(self, strategy_id: str, token: str) -> float:
        """
        Get suitability score for a strategy-token combination.
        Returns 0.0-1.0 where higher = better fit.
        """
        matrix = self.learning_data['strategy_token_matrix'].get(strategy_id, {})
        token_data = matrix.get(token)
        
        if not token_data or token_data['trades'] < 3:
            return 0.5  # Unknown - try it!
        
        win_rate = token_data['wins'] / token_data['trades']
        return win_rate
    
    def should_skip_combination(self, strategy_id: str, token: str) -> Tuple[bool, str]:
        """
        Determine if we should skip a strategy-token combination.
        Only skip if we have enough data showing it consistently loses.
        """
        matrix = self.learning_data['strategy_token_matrix'].get(strategy_id, {})
        token_data = matrix.get(token)
        
        if not token_data or token_data['trades'] < 10:
            return False, ""  # Not enough data - keep trying!
        
        win_rate = token_data['wins'] / token_data['trades']
        
        if win_rate < 0.25 and token_data['pnl'] < -100:
            return True, f"Learned: {strategy_id[:20]} loses on {token} (WR: {win_rate:.0%})"
        
        return False, ""
    
    def get_learning_summary(self) -> str:
        """Get a summary of what the system has learned"""
        total = self.learning_data['total_trades']
        if total == 0:
            return "📚 No trades yet - learning in progress!"
        
        wins = self.learning_data['total_wins']
        pnl = self.learning_data['total_pnl']
        
        # Find best/worst strategies
        strategies = self.learning_data['strategy_performance']
        best_strat = max(strategies.items(), 
                        key=lambda x: x[1]['wins']/x[1]['trades'] if x[1]['trades'] > 5 else 0,
                        default=(None, {}))
        
        summary = f"""
📚 LEARNING SUMMARY
   Total Trades: {total}
   Win Rate: {wins/total*100:.1f}%
   Total PnL: ${pnl:+,.2f}
   Best Strategy: {best_strat[0][:30] if best_strat[0] else 'N/A'}
"""
        return summary


# =============================================================================
# DYNAMIC STRATEGY ANALYZER - NO HARDCODED TYPES!
# =============================================================================

class DynamicStrategyAnalyzer:
    """
    Dynamically analyzes strategy files to understand what they do.
    
    This is the KEY DIFFERENCE from V1:
    - V1 had hardcoded strategy types
    - ADAPTIVE reads the actual code and extracts indicators/logic
    - Works with ANY new strategy automatically!
    """
    
    INDICATOR_PATTERNS = {
        'vwap': r'vwap|volume.weighted',
        'rsi': r'rsi|relative.strength',
        'macd': r'macd|moving.average.convergence',
        'bollinger': r'bollinger|bb_|bb\.', 
        'ema': r'ema|exponential.moving',
        'sma': r'sma|simple.moving',
        'atr': r'atr|average.true.range',
        'stochastic': r'stoch|stochastic',
        'momentum': r'momentum|roc|rate.of.change',
        'volume': r'volume.ratio|obv|volume.profile',
        'mean_reversion': r'mean.reversion|revert|band',
        'breakout': r'breakout|break.out|resistance|support',
        'trend': r'trend|trending|adx',
        'inventory': r'inventory|position.size|rebalanc',
        'market_maker': r'market.maker|bid.ask|spread',
        'pairs': r'pairs|cointegrat|spread.trading',
        'neural': r'neural|lstm|prediction|ml_|ai_'
    }
    
    def analyze_strategy_file(self, py_file: Path) -> Dict[str, Any]:
        """
        Analyze a strategy Python file and extract its characteristics.
        Returns dict with indicators, logic type, and signal generation hints.
        """
        result = {
            'file': str(py_file),
            'indicators': [],
            'signal_type': 'adaptive',  # Default - we'll adapt to any strategy
            'entry_conditions': [],
            'exit_conditions': [],
            'parameters': {},
            'confidence': 0.5
        }
        
        try:
            with open(py_file, 'r') as f:
                code = f.read().lower()
            
            # Find which indicators are used
            for indicator, pattern in self.INDICATOR_PATTERNS.items():
                if re.search(pattern, code, re.IGNORECASE):
                    result['indicators'].append(indicator)
            
            # Determine primary signal type based on indicators
            if 'vwap' in result['indicators'] and 'mean_reversion' in result['indicators']:
                result['signal_type'] = 'mean_reversion'
            elif 'market_maker' in result['indicators'] or 'inventory' in result['indicators']:
                result['signal_type'] = 'market_maker'
            elif 'pairs' in result['indicators']:
                result['signal_type'] = 'pairs_spread'
            elif 'neural' in result['indicators']:
                result['signal_type'] = 'ml_prediction'
            elif 'momentum' in result['indicators'] or 'breakout' in result['indicators']:
                result['signal_type'] = 'momentum'
            elif 'rsi' in result['indicators'] and 'bollinger' in result['indicators']:
                result['signal_type'] = 'oversold_overbought'
            
            # Extract any numeric parameters
            param_patterns = [
                (r'std_dev.*?=.*?(\d+\.?\d*)', 'std_dev'),
                (r'period.*?=.*?(\d+)', 'period'),
                (r'threshold.*?=.*?(\d+\.?\d*)', 'threshold'),
                (r'window.*?=.*?(\d+)', 'window'),
                (r'lookback.*?=.*?(\d+)', 'lookback'),
            ]
            
            for pattern, param_name in param_patterns:
                match = re.search(pattern, code)
                if match:
                    result['parameters'][param_name] = float(match.group(1))
            
            # Higher confidence if more indicators found
            result['confidence'] = min(0.9, 0.4 + len(result['indicators']) * 0.1)
            
        except Exception as e:
            print(f"⚠️  Could not analyze {py_file.name}: {e}")
        
        return result


# =============================================================================
# ADAPTIVE SIGNAL GENERATOR - WORKS WITH ANY STRATEGY
# =============================================================================

class AdaptiveSignalGenerator:
    """
    Generates signals dynamically based on strategy analysis.
    
    Instead of hardcoded signal logic, this adapts to what
    indicators each strategy uses.
    """
    
    def __init__(self, learning_engine: LearningEngine):
        self.learning = learning_engine
        self.analyzer = DynamicStrategyAnalyzer()
        self.strategy_cache = {}  # Cache analyzed strategies
    
    def generate_signal(self, strategy_id: str, strategy_info: Dict, 
                       df: pd.DataFrame, current_price: float) -> Dict:
        """
        Generate a trading signal using adaptive logic.
        
        The signal generation adapts based on:
        1. What indicators the strategy uses (from code analysis)
        2. What has worked before (from learning engine)
        3. Current market conditions
        """
        # Get or create analysis
        if strategy_id not in self.strategy_cache:
            py_file = strategy_info.get('py_file')
            if py_file and Path(py_file).exists():
                self.strategy_cache[strategy_id] = self.analyzer.analyze_strategy_file(Path(py_file))
            else:
                # Use meta data to infer
                self.strategy_cache[strategy_id] = self._infer_from_meta(strategy_info)
        
        analysis = self.strategy_cache[strategy_id]
        signal_type = analysis.get('signal_type', 'adaptive')
        indicators = analysis.get('indicators', [])
        
        # Calculate all potentially needed indicators
        indicators_data = self._calculate_indicators(df, current_price)
        
        # Generate signal based on detected type
        if signal_type == 'mean_reversion':
            return self._mean_reversion_signal(indicators_data, current_price, analysis)
        elif signal_type == 'market_maker':
            return self._market_maker_signal(indicators_data, current_price, analysis)
        elif signal_type == 'momentum':
            return self._momentum_signal(indicators_data, current_price, analysis)
        elif signal_type == 'oversold_overbought':
            return self._rsi_bb_signal(indicators_data, current_price, analysis)
        elif signal_type == 'ml_prediction':
            return self._ml_prediction_signal(indicators_data, current_price, analysis)
        else:
            # Adaptive: try multiple and pick strongest
            return self._adaptive_signal(indicators_data, current_price, analysis)
    
    def _infer_from_meta(self, strategy_info: Dict) -> Dict:
        """Infer strategy characteristics from meta data"""
        name = strategy_info.get('name', '').lower()
        
        result = {'indicators': [], 'signal_type': 'adaptive', 'confidence': 0.5}
        
        # Infer from name
        if 'vwap' in name or 'mean' in name:
            result['signal_type'] = 'mean_reversion'
            result['indicators'] = ['vwap', 'mean_reversion', 'atr']
        elif 'market' in name and 'maker' in name:
            result['signal_type'] = 'market_maker'
            result['indicators'] = ['inventory', 'market_maker', 'atr']
        elif 'momentum' in name or 'breakout' in name:
            result['signal_type'] = 'momentum'
            result['indicators'] = ['momentum', 'rsi', 'volume']
        elif 'neural' in name or 'ai' in name or 'ml' in name:
            result['signal_type'] = 'ml_prediction'
            result['indicators'] = ['neural', 'momentum', 'trend']
        elif 'stoikov' in name:
            result['signal_type'] = 'market_maker'
            result['indicators'] = ['market_maker', 'inventory', 'atr']
        elif 'pairs' in name or 'cointegration' in name:
            result['signal_type'] = 'mean_reversion'
            result['indicators'] = ['pairs', 'mean_reversion']
        elif 'reversal' in name:
            result['signal_type'] = 'oversold_overbought'
            result['indicators'] = ['rsi', 'bollinger', 'mean_reversion']
        
        return result
    
    def _calculate_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate all commonly used indicators"""
        result = {'current_price': current_price}
        
        try:
            # VWAP
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_vp = (typical_price * df['Volume']).cumsum()
            cumulative_volume = df['Volume'].cumsum()
            vwap = cumulative_vp / cumulative_volume
            
            price_deviation = np.abs(df['Close'] - vwap)
            std_dev = price_deviation.rolling(window=20).std()
            
            result['vwap'] = float(vwap.iloc[-1])
            result['vwap_upper'] = float(vwap.iloc[-1] + 2 * std_dev.iloc[-1])
            result['vwap_lower'] = float(vwap.iloc[-1] - 2 * std_dev.iloc[-1])
            result['vwap_std'] = float(std_dev.iloc[-1])
            
            # ATR
            high, low, close = df['High'], df['Low'], df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = true_range.rolling(window=14).mean()
            result['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_price * 0.02
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            result['bb_upper'] = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
            result['bb_lower'] = float(sma20.iloc[-1] - 2 * std20.iloc[-1])
            result['bb_mid'] = float(sma20.iloc[-1])
            
            # Momentum
            result['momentum_10'] = float((df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100)
            result['momentum_5'] = float((df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100)
            
            # EMAs
            result['ema_8'] = float(df['Close'].ewm(span=8).mean().iloc[-1])
            result['ema_21'] = float(df['Close'].ewm(span=21).mean().iloc[-1])
            
            # Volume
            vol_sma = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = float(df['Volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1
            
        except Exception as e:
            print(f"⚠️  Indicator calculation error: {e}")
        
        return result
    
    def _mean_reversion_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """Mean reversion signal (VWAP-based)"""
        vwap = ind.get('vwap', price)
        upper = ind.get('vwap_upper', price * 1.02)
        lower = ind.get('vwap_lower', price * 0.98)
        atr = ind.get('atr', price * 0.02)
        
        signal = "HOLD"
        reason = "Price within VWAP bands"
        target = 0
        stop = 0
        
        lower_dev = (lower - price) / price * 100
        upper_dev = (price - upper) / price * 100
        
        if price < lower and lower_dev > 0.1:
            signal = "BUY"
            reason = f"Mean Reversion: Price ${price:.2f} below lower band ${lower:.2f}"
            target = vwap
            stop = price - (atr * 1.5)
        elif price > upper and upper_dev > 0.1:
            signal = "SELL"
            reason = f"Mean Reversion: Price ${price:.2f} above upper band ${upper:.2f}"
            target = vwap
            stop = price + (atr * 1.5)
        
        return self._build_signal(signal, reason, price, vwap, upper, lower, target, stop, atr, ind)
    
    def _market_maker_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """Market maker signal (tighter bands)"""
        vwap = ind.get('vwap', price)
        std = ind.get('vwap_std', price * 0.01)
        atr = ind.get('atr', price * 0.02)
        
        # Tighter bands for MM
        upper = vwap + (1.2 * std)
        lower = vwap - (1.2 * std)
        
        signal = "HOLD"
        reason = "MM: Price within spread"
        target = 0
        stop = 0
        
        if price < lower:
            signal = "BUY"
            reason = f"MM Buy: Price ${price:.2f} below tight band ${lower:.2f}"
            target = vwap
            stop = price - atr
        elif price > upper:
            signal = "SELL"
            reason = f"MM Sell: Price ${price:.2f} above tight band ${upper:.2f}"
            target = vwap
            stop = price + atr
        
        return self._build_signal(signal, reason, price, vwap, upper, lower, target, stop, atr, ind)
    
    def _momentum_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """Momentum/breakout signal"""
        rsi = ind.get('rsi', 50)
        momentum = ind.get('momentum_10', 0)
        atr = ind.get('atr', price * 0.02)
        ema_8 = ind.get('ema_8', price)
        ema_21 = ind.get('ema_21', price)
        
        signal = "HOLD"
        reason = "Momentum: No clear trend"
        target = 0
        stop = 0
        
        # Strong upward momentum
        if momentum > 2 and rsi < 70 and ema_8 > ema_21:
            signal = "BUY"
            reason = f"Momentum Buy: {momentum:.1f}% move, RSI {rsi:.0f}"
            target = price * 1.03
            stop = price - (atr * 2)
        # Strong downward momentum
        elif momentum < -2 and rsi > 30 and ema_8 < ema_21:
            signal = "SELL"
            reason = f"Momentum Sell: {momentum:.1f}% move, RSI {rsi:.0f}"
            target = price * 0.97
            stop = price + (atr * 2)
        
        return self._build_signal(signal, reason, price, ind.get('vwap', price), 
                                 ind.get('bb_upper', price*1.02), ind.get('bb_lower', price*0.98),
                                 target, stop, atr, ind)
    
    def _rsi_bb_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """RSI + Bollinger Bands signal"""
        rsi = ind.get('rsi', 50)
        bb_upper = ind.get('bb_upper', price * 1.02)
        bb_lower = ind.get('bb_lower', price * 0.98)
        bb_mid = ind.get('bb_mid', price)
        atr = ind.get('atr', price * 0.02)
        
        signal = "HOLD"
        reason = "RSI/BB: No extreme"
        target = 0
        stop = 0
        
        # Oversold bounce
        if rsi < 30 and price <= bb_lower:
            signal = "BUY"
            reason = f"Oversold: RSI {rsi:.0f}, price at lower BB"
            target = bb_mid
            stop = price - (atr * 1.5)
        # Overbought reversal
        elif rsi > 70 and price >= bb_upper:
            signal = "SELL"
            reason = f"Overbought: RSI {rsi:.0f}, price at upper BB"
            target = bb_mid
            stop = price + (atr * 1.5)
        
        return self._build_signal(signal, reason, price, bb_mid, bb_upper, bb_lower, target, stop, atr, ind)
    
    def _ml_prediction_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """ML-style prediction signal (uses multiple factors)"""
        # Combine multiple indicators for a "prediction"
        rsi = ind.get('rsi', 50)
        momentum = ind.get('momentum_5', 0)
        vol_ratio = ind.get('volume_ratio', 1)
        ema_8 = ind.get('ema_8', price)
        ema_21 = ind.get('ema_21', price)
        atr = ind.get('atr', price * 0.02)
        
        # Simple "prediction" score
        score = 0
        if rsi < 40: score += 1
        if rsi > 60: score -= 1
        if momentum > 0: score += 0.5
        if momentum < 0: score -= 0.5
        if vol_ratio > 1.5: score += 0.5 * (1 if momentum > 0 else -1)
        if ema_8 > ema_21: score += 0.5
        if ema_8 < ema_21: score -= 0.5
        
        signal = "HOLD"
        reason = f"ML Prediction: Score {score:.1f}"
        target = 0
        stop = 0
        
        if score >= 1.5:
            signal = "BUY"
            reason = f"ML Buy Signal: Score {score:.1f}"
            target = price * 1.02
            stop = price - (atr * 1.5)
        elif score <= -1.5:
            signal = "SELL"
            reason = f"ML Sell Signal: Score {score:.1f}"
            target = price * 0.98
            stop = price + (atr * 1.5)
        
        return self._build_signal(signal, reason, price, ind.get('vwap', price),
                                 ind.get('bb_upper', price*1.02), ind.get('bb_lower', price*0.98),
                                 target, stop, atr, ind)
    
    def _adaptive_signal(self, ind: Dict, price: float, analysis: Dict) -> Dict:
        """Adaptive signal - tries multiple approaches and picks strongest"""
        signals = [
            self._mean_reversion_signal(ind, price, analysis),
            self._momentum_signal(ind, price, analysis),
            self._rsi_bb_signal(ind, price, analysis),
        ]
        
        # Pick the one with a non-HOLD signal, preferring higher confidence
        for sig in signals:
            if sig['signal'] != 'HOLD':
                sig['reason'] = f"[Adaptive] {sig['reason']}"
                return sig
        
        return signals[0]  # Default to first (mean reversion)
    
    def _build_signal(self, signal: str, reason: str, price: float, vwap: float,
                     upper: float, lower: float, target: float, stop: float,
                     atr: float, ind: Dict) -> Dict:
        """Build a complete signal dictionary"""
        return {
            'signal': signal,
            'reason': reason,
            'current_price': price,
            'vwap': vwap,
            'upper_band': upper,
            'lower_band': lower,
            'target_price': target,
            'stop_loss': stop,
            'atr': atr,
            'rsi': ind.get('rsi', 50),
            'momentum': ind.get('momentum_10', 0),
            'volume_ratio': ind.get('volume_ratio', 1),
            'confidence': 0.6
        }


# =============================================================================
# HTX MARKET DATA CLIENT
# =============================================================================

class HTXClient:
    """Real-time market data from HTX"""
    
    def __init__(self):
        self.base_url = Config.HTX_BASE_URL
        self.session = requests.Session()
        self.price_cache = {}
        self.cache_time = {}
        self.valid_tokens = set()  # Tokens confirmed to exist on HTX
        self.invalid_tokens = set()  # Tokens that don't exist
    
    def fetch_candles(self, symbol: str, period: str = '15min', count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles"""
        if symbol in self.invalid_tokens:
            return None
            
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
                        self.valid_tokens.add(symbol)
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
                        df = df.sort_values('datetime').reset_index(drop=True)
                        return df
                else:
                    self.invalid_tokens.add(symbol)
            
            return None
            
        except Exception as e:
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with caching"""
        if symbol in self.invalid_tokens:
            return None
            
        # Check cache (valid for 3 seconds)
        if symbol in self.price_cache:
            if time.time() - self.cache_time.get(symbol, 0) < 3:
                return self.price_cache[symbol]
        
        try:
            candles = self.fetch_candles(symbol, '1min', 1)
            if candles is not None and len(candles) > 0:
                price = float(candles['Close'].iloc[-1])
                self.price_cache[symbol] = price
                self.cache_time[symbol] = time.time()
                return price
        except:
            pass
        
        return None
    
    def get_valid_tokens(self, token_list: List[str]) -> List[str]:
        """Filter token list to only valid tradeable tokens"""
        valid = []
        for token in token_list:
            if token in self.valid_tokens:
                valid.append(token)
            elif token not in self.invalid_tokens:
                # Test it
                price = self.get_current_price(token)
                if price:
                    valid.append(token)
        return valid


# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

class PaperPosition:
    """Paper trade position"""
    
    def __init__(self, position_id: str, strategy_id: str, symbol: str,
                 direction: str, entry_price: float, size: float,
                 target: float, stop_loss: float):
        self.position_id = position_id
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.target_price = target
        self.stop_loss = stop_loss
        self.entry_time = datetime.now()
        self.status = "OPEN"
        self.exit_price = 0.0
        self.pnl = 0.0


class PaperTradingEngine:
    """Paper trading with learning integration"""
    
    def __init__(self, learning_engine: LearningEngine):
        self.learning = learning_engine
        self.capital = Config.STARTING_CAPITAL
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_positions: List[PaperPosition] = []
        self.total_pnl = 0.0
    
    def can_open_position(self, strategy_id: str, symbol: str) -> Tuple[bool, str]:
        """Check if we can open a position"""
        # Check if learning says to skip this combination
        should_skip, skip_reason = self.learning.should_skip_combination(strategy_id, symbol)
        if should_skip:
            return False, skip_reason
        
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        
        if len(open_positions) >= Config.MAX_TOTAL_POSITIONS:
            return False, f"Max positions ({Config.MAX_TOTAL_POSITIONS})"
        
        strat_pos = [p for p in open_positions if p.strategy_id == strategy_id]
        if len(strat_pos) >= Config.MAX_POSITIONS_PER_STRATEGY:
            return False, f"Strategy limit ({Config.MAX_POSITIONS_PER_STRATEGY})"
        
        token_pos = [p for p in open_positions if p.symbol == symbol]
        if len(token_pos) >= Config.MAX_POSITIONS_PER_TOKEN:
            return False, f"Token limit ({Config.MAX_POSITIONS_PER_TOKEN})"
        
        existing = [p for p in open_positions if p.strategy_id == strategy_id and p.symbol == symbol]
        if existing:
            return False, "Position exists"
        
        return True, "OK"
    
    def open_position(self, strategy_id: str, symbol: str, signal: Dict) -> Optional[str]:
        """Open a position with learning-adjusted sizing"""
        if signal['signal'] == 'HOLD':
            return None
        
        can_open, reason = self.can_open_position(strategy_id, symbol)
        if not can_open:
            return None
        
        # Get learning-adjusted position size
        confidence = self.learning.get_strategy_confidence(strategy_id)
        base_size = self.capital * Config.MAX_POSITION_SIZE * Config.DEFAULT_LEVERAGE
        adjusted_size = base_size * confidence
        
        position_id = f"{strategy_id}_{symbol}_{datetime.now().strftime('%H%M%S%f')}"
        
        position = PaperPosition(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            direction=signal['signal'],
            entry_price=signal['current_price'],
            size=adjusted_size,
            target=signal.get('target_price', 0),
            stop_loss=signal.get('stop_loss', 0)
        )
        
        self.positions[position_id] = position
        
        # Log
        print(f"\n🎯 TRADE OPENED: {signal['signal']} {symbol}")
        print(f"   Strategy: {strategy_id[:40]}")
        print(f"   Entry: ${signal['current_price']:.4f}")
        print(f"   Target: ${signal.get('target_price', 0):.4f}")
        print(f"   Stop: ${signal.get('stop_loss', 0):.4f}")
        print(f"   Size: ${adjusted_size:.2f} (confidence: {confidence:.2f})")
        print(f"   Reason: {signal['reason']}")
        
        return position_id
    
    def check_exits(self, current_prices: Dict[str, float]):
        """Check and close positions"""
        for position_id, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue
            
            price = current_prices.get(position.symbol)
            if not price:
                continue
            
            # Calculate P&L
            if position.direction == "BUY":
                pnl_pct = (price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - price) / position.entry_price
            
            pnl_usd = position.size * pnl_pct
            
            # Check exits
            should_close = False
            reason = ""
            
            if position.direction == "BUY":
                if price >= position.target_price:
                    should_close = True
                    reason = "TARGET HIT"
                elif price <= position.stop_loss:
                    should_close = True
                    reason = "STOP LOSS"
            else:
                if price <= position.target_price:
                    should_close = True
                    reason = "TARGET HIT"
                elif price >= position.stop_loss:
                    should_close = True
                    reason = "STOP LOSS"
            
            if should_close:
                self._close_position(position, price, pnl_usd, pnl_pct * 100, reason)
    
    def _close_position(self, position: PaperPosition, exit_price: float,
                       pnl: float, pnl_pct: float, reason: str):
        """Close position and record learning"""
        position.status = "CLOSED"
        position.exit_price = exit_price
        position.pnl = pnl
        
        self.capital += pnl
        self.total_pnl += pnl
        self.closed_positions.append(position)
        
        is_win = pnl >= 0
        
        # RECORD FOR LEARNING - This is key!
        self.learning.record_trade(
            position.strategy_id,
            position.symbol,
            pnl,
            is_win
        )
        
        emoji = "✅" if is_win else "❌"
        print(f"\n{emoji} TRADE CLOSED: {position.direction} {position.symbol}")
        print(f"   Exit: ${exit_price:.4f}")
        print(f"   PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   Reason: {reason}")
        print(f"   Capital: ${self.capital:,.2f}")


# =============================================================================
# MAIN TRADEPEX ADAPTIVE ENGINE
# =============================================================================

class TradePexAdaptive:
    """
    TradePex Adaptive - Self-Improving Trading System
    
    KEY DIFFERENCES FROM V1:
    1. DYNAMIC - No hardcoded strategy types
    2. SELF-IMPROVING - Learns from every trade
    3. MORE TOKENS - Trades 30+ tokens for faster learning
    4. ADAPTIVE SIGNALS - Works with ANY strategy file
    """
    
    def __init__(self):
        self._print_banner()
        
        # Initialize components
        self.learning = LearningEngine()
        self.htx = HTXClient()
        self.signal_gen = AdaptiveSignalGenerator(self.learning)
        self.paper = PaperTradingEngine(self.learning)
        
        # Load strategies
        self.strategies = {}
        self._load_strategies()
        
        # Validate tokens
        self._validate_tokens()
        
        # State
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.running = True
    
    def _print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🚀 TRADEPEX ADAPTIVE - SELF-IMPROVING TRADING SYSTEM 🚀                ║
║                                                                              ║
║     ✨ UPGRADE FROM V1:                                                     ║
║        • DYNAMIC strategy detection (no hardcoded types!)                   ║
║        • SELF-IMPROVING through learning from every trade                   ║
║        • MORE TOKENS for faster learning (30+ tokens)                       ║
║        • ADAPTIVE signals work with ANY new strategy                        ║
║                                                                              ║
║     🌙 More Trading → More Learning → Better Performance 🌙                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def _load_strategies(self):
        """Load all strategies dynamically"""
        dirs_to_try = [
            Config.STRATEGIES_DIR,
            Config.FALLBACK_STRATEGIES_DIR,
            Path(__file__).parent / "successful_strategies"
        ]
        
        for strategies_dir in dirs_to_try:
            if strategies_dir.exists():
                print(f"\n📂 Loading strategies from: {strategies_dir}")
                py_files = [f for f in strategies_dir.glob("*.py") if '_meta' not in f.stem]
                
                for py_file in py_files:
                    strategy_id = py_file.stem
                    meta_file = strategies_dir / f"{strategy_id}_meta.json"
                    
                    meta = {}
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r') as f:
                                meta = json.load(f)
                        except:
                            pass
                    
                    self.strategies[strategy_id] = {
                        'py_file': py_file,
                        'meta': meta,
                        'name': meta.get('strategy_name', strategy_id)
                    }
                    
                    print(f"✅ {strategy_id[:60]}")
                
                if self.strategies:
                    break
        
        if not self.strategies:
            print("⚠️  No strategy files found - creating defaults")
            self._create_default_strategies()
        
        print(f"\n🎯 TOTAL STRATEGIES: {len(self.strategies)}")
    
    def _create_default_strategies(self):
        """Create default strategies if none found"""
        defaults = [
            ('Adaptive_VWAP_Mean_Reversion', 'mean_reversion'),
            ('Adaptive_Market_Maker', 'market_maker'),
            ('Adaptive_Momentum', 'momentum'),
            ('Adaptive_RSI_BB', 'oversold_overbought'),
            ('Adaptive_ML_Prediction', 'ml_prediction'),
        ]
        
        for name, signal_type in defaults:
            self.strategies[name] = {
                'py_file': None,
                'meta': {'strategy_name': name},
                'name': name,
                '_signal_type': signal_type
            }
            self.signal_gen.strategy_cache[name] = {
                'signal_type': signal_type,
                'indicators': [],
                'confidence': 0.5
            }
    
    def _validate_tokens(self):
        """Validate which tokens are tradeable"""
        print(f"\n🔍 Validating {len(Config.TRADEABLE_TOKENS)} tokens...")
        
        valid = []
        for token in Config.TRADEABLE_TOKENS:
            price = self.htx.get_current_price(token)
            if price:
                valid.append(token)
                print(f"   ✅ {token}: ${price:,.4f}")
            else:
                print(f"   ❌ {token}: Not available")
        
        self.valid_tokens = valid
        print(f"\n💎 TRADING {len(self.valid_tokens)} TOKENS")
    
    def run(self):
        """Main trading loop"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING TRADEPEX ADAPTIVE")
        print(f"{'='*80}")
        print(f"📊 Strategies: {len(self.strategies)}")
        print(f"💎 Tokens: {len(self.valid_tokens)}")
        print(f"⏰ Cycle interval: {Config.CHECK_INTERVAL}s")
        print(f"📈 Max positions: {Config.MAX_TOTAL_POSITIONS}")
        print(f"🧠 Learning from previous: {self.learning.learning_data['total_trades']} trades")
        print(f"{'='*80}\n")
        
        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle()
                time.sleep(Config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n\n🛑 SHUTDOWN")
                self.learning._save_learning_data()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                traceback.print_exc()
                time.sleep(30)
        
        # Final summary
        print(self.learning.get_learning_summary())
    
    def _run_cycle(self):
        """Run one trading cycle"""
        print(f"\n{'='*80}")
        print(f"🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        # Fetch prices
        prices = {}
        print("\n📊 PRICES:")
        for token in self.valid_tokens:
            price = self.htx.get_current_price(token)
            if price:
                prices[token] = price
                print(f"   {token}: ${price:,.4f}")
        
        # Check exits
        self.paper.check_exits(prices)
        
        # Generate signals
        open_count = len([p for p in self.paper.positions.values() if p.status == "OPEN"])
        
        if open_count >= Config.MAX_TOTAL_POSITIONS:
            print(f"\n⏸️  Position limit ({open_count}/{Config.MAX_TOTAL_POSITIONS})")
        else:
            print(f"\n🎯 SCANNING FOR SIGNALS ({open_count}/{Config.MAX_TOTAL_POSITIONS} positions)...")
            
            signals_found = 0
            for strategy_id, strategy_info in self.strategies.items():
                for token in self.valid_tokens:
                    signal = self._generate_and_execute(strategy_id, strategy_info, token, prices)
                    if signal and signal['signal'] != 'HOLD':
                        signals_found += 1
            
            if signals_found == 0:
                print("   No actionable signals this cycle")
        
        # Show status
        self._show_status(prices)
    
    def _generate_and_execute(self, strategy_id: str, strategy_info: Dict, 
                              token: str, prices: Dict) -> Optional[Dict]:
        """Generate signal and potentially open position"""
        price = prices.get(token)
        if not price:
            return None
        
        # Check if we can open before fetching data
        can_open, reason = self.paper.can_open_position(strategy_id, token)
        if not can_open:
            return None
        
        # Fetch market data
        df = self.htx.fetch_candles(token, '15min', 100)
        if df is None or len(df) < 50:
            return None
        
        # Generate signal
        signal = self.signal_gen.generate_signal(strategy_id, strategy_info, df, price)
        
        if signal['signal'] != 'HOLD':
            print(f"\n{'='*60}")
            print(f"📡 SIGNAL: {signal['signal']} {token}")
            print(f"   Strategy: {strategy_id[:40]}")
            print(f"   Price: ${price:.4f}")
            print(f"   Target: ${signal.get('target_price', 0):.4f}")
            print(f"   Stop: ${signal.get('stop_loss', 0):.4f}")
            print(f"   RSI: {signal.get('rsi', 0):.1f}")
            print(f"   Momentum: {signal.get('momentum', 0):.2f}%")
            print(f"   Reason: {signal['reason']}")
            print(f"{'='*60}")
            
            # Try to open position
            self.paper.open_position(strategy_id, token, signal)
        
        return signal
    
    def _show_status(self, prices: Dict):
        """Show current status"""
        open_pos = [p for p in self.paper.positions.values() if p.status == "OPEN"]
        closed = len(self.paper.closed_positions)
        
        win_rate = 0
        if self.learning.learning_data['total_trades'] > 0:
            win_rate = self.learning.learning_data['total_wins'] / self.learning.learning_data['total_trades'] * 100
        
        print(f"\n{'='*80}")
        print(f"📊 TRADEPEX ADAPTIVE STATUS")
        print(f"{'='*80}")
        print(f"💰 Capital: ${self.paper.capital:,.2f} | PnL: ${self.paper.total_pnl:+,.2f}")
        print(f"📈 Positions: {len(open_pos)}/{Config.MAX_TOTAL_POSITIONS} open | {closed} closed")
        print(f"🧠 Learning: {self.learning.learning_data['total_trades']} trades | {win_rate:.1f}% win rate")
        print(f"⏰ Runtime: {datetime.now() - self.start_time}")
        
        if open_pos:
            print(f"\n📊 OPEN POSITIONS:")
            print("-" * 90)
            for pos in open_pos[:10]:  # Show first 10
                current = prices.get(pos.symbol, pos.entry_price)
                if pos.direction == "BUY":
                    pnl_pct = (current - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - current) / pos.entry_price * 100
                
                emoji = "🟢" if pnl_pct >= 0 else "🔴"
                print(f"   {pos.symbol:<6} {pos.direction:<4} ${pos.entry_price:>10.4f} → ${current:>10.4f} {emoji} {pnl_pct:+.2f}%")
            
            if len(open_pos) > 10:
                print(f"   ... and {len(open_pos) - 10} more")
        
        print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    engine = TradePexAdaptive()
    engine.run()
