#!/usr/bin/env python3
"""
TRADEPEX 24/7 LIVE PAPER TRADING - PROPER POSITION MANAGEMENT
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Dict, List, Optional
import logging

# =============================================================================
# CONFIGURATION - PROPER RISK MANAGEMENT
# =============================================================================

class Config:
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.2  # 20% per trade (REDUCED)
    DEFAULT_LEVERAGE = 3
    HTX_BASE_URL = "https://api.huobi.pro"
    TRADEABLE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX']
    STRATEGIES_DIR = Path("/root/KEEP_SAFE/v1/APEX/successful_strategies")
    CHECK_INTERVAL = 30

    # PROPER POSITION LIMITS
    MAX_TOTAL_POSITIONS = 8           # Max 8 positions total
    MAX_POSITIONS_PER_STRATEGY = 2    # Max 2 positions per strategy
    MAX_POSITIONS_PER_TOKEN = 2       # Max 2 positions per token

# =============================================================================
# REAL HTX CLIENT
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
# STRATEGY EXECUTION ENGINE
# =============================================================================

class StrategyExecutor:
    def calculate_vwap(self, df: pd.DataFrame) -> Dict:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_vp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        vwap = cumulative_vp / cumulative_volume

        price_deviation = np.abs(df['Close'] - vwap)
        std_dev = price_deviation.rolling(window=20).std()

        upper_band = vwap + (2.0 * std_dev)
        lower_band = vwap - (2.0 * std_dev)

        return {
            'vwap': float(vwap.iloc[-1]),
            'upper_band': float(upper_band.iloc[-1]),
            'lower_band': float(lower_band.iloc[-1]),
            'std_dev': float(std_dev.iloc[-1])
        }

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    def execute_vwap_strategy(self, df: pd.DataFrame, current_price: float) -> Dict:
        vwap_data = self.calculate_vwap(df)
        atr = self.calculate_atr(df)

        signal = "HOLD"
        reason = "Price within bands"
        target_price = 0.0
        stop_loss = 0.0

        # Only trade if we have sufficient deviation
        lower_band_distance = (vwap_data['lower_band'] - current_price) / current_price * 100
        upper_band_distance = (current_price - vwap_data['upper_band']) / current_price * 100

        if current_price < vwap_data['lower_band'] and lower_band_distance > 0.1:  # Min 0.1% deviation
            signal = "BUY"
            reason = f"Price ${current_price:.2f} below VWAP lower band ${vwap_data['lower_band']:.2f} (dev: {lower_band_distance:.2f}%)"
            target_price = vwap_data['vwap']
            stop_loss = current_price - (atr * 1.5)

        elif current_price > vwap_data['upper_band'] and upper_band_distance > 0.1:  # Min 0.1% deviation
            signal = "SELL"
            reason = f"Price ${current_price:.2f} above VWAP upper band ${vwap_data['upper_band']:.2f} (dev: {upper_band_distance:.2f}%)"
            target_price = vwap_data['vwap']
            stop_loss = current_price + (atr * 1.5)

        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'vwap': vwap_data['vwap'],
            'upper_band': vwap_data['upper_band'],
            'lower_band': vwap_data['lower_band'],
            'target_price': target_price,
            'stop_loss': stop_loss,
            'atr': atr
        }

# =============================================================================
# PAPER TRADING ENGINE - PROPER POSITION MANAGEMENT
# =============================================================================

class PaperTrade:
    def __init__(self, strategy_id: str, symbol: str, direction: str,
                 entry_price: float, size: float, target: float, stop_loss: float):
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
    def __init__(self):
        self.positions = {}
        self.capital = Config.STARTING_CAPITAL
        self.trade_history = []

    def can_open_position(self, strategy_id: str, symbol: str) -> tuple[bool, str]:
        """CHECK ALL POSITION LIMITS BEFORE OPENING"""

        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]

        # 1. Check total position limit
        if len(open_positions) >= Config.MAX_TOTAL_POSITIONS:
            return False, f"Max total positions ({Config.MAX_TOTAL_POSITIONS}) reached"

        # 2. Check strategy position limit
        strategy_positions = [p for p in open_positions if p.strategy_id == strategy_id]
        if len(strategy_positions) >= Config.MAX_POSITIONS_PER_STRATEGY:
            return False, f"Strategy {strategy_id} has max positions ({Config.MAX_POSITIONS_PER_STRATEGY})"

        # 3. Check token position limit
        token_positions = [p for p in open_positions if p.symbol == symbol]
        if len(token_positions) >= Config.MAX_POSITIONS_PER_TOKEN:
            return False, f"Token {symbol} has max positions ({Config.MAX_POSITIONS_PER_TOKEN})"

        # 4. Check if we already have this exact strategy/token position
        existing_position = [p for p in open_positions if p.strategy_id == strategy_id and p.symbol == symbol]
        if existing_position:
            return False, f"Already have {strategy_id} position on {symbol}"

        return True, "OK"

    def open_position(self, strategy_id: str, symbol: str, signal: Dict, size_percent: float = 0.15):
        """OPEN POSITION WITH PROPER CHECKS"""

        if signal['signal'] == 'HOLD':
            return None

        # Check if we can open this position
        can_open, reason = self.can_open_position(strategy_id, symbol)
        if not can_open:
            print(f"⏸️  POSITION BLOCKED: {strategy_id} {symbol} - {reason}")
            return None

        position_id = f"{strategy_id}_{symbol}_{datetime.now().strftime('%H%M%S')}"
        size_usd = self.capital * size_percent
        leverage_size = size_usd * Config.DEFAULT_LEVERAGE

        position = PaperTrade(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=signal['signal'],
            entry_price=signal['current_price'],
            size=leverage_size,
            target=signal.get('target_price', 0),
            stop_loss=signal.get('stop_loss', 0)
        )

        self.positions[position_id] = position

        trade_log = {
            'timestamp': datetime.now(),
            'position_id': position_id,
            'strategy': strategy_id,
            'symbol': symbol,
            'action': 'OPEN',
            'direction': signal['signal'],
            'entry_price': signal['current_price'],
            'size': leverage_size,
            'target': signal.get('target_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'reason': signal['reason']
        }

        self.trade_history.append(trade_log)
        print(f"🎯 OPENED: {position_id} | {signal['signal']} {symbol} @ ${signal['current_price']:.2f}")
        print(f"   Target: ${signal.get('target_price', 0):.2f} | Stop: ${signal.get('stop_loss', 0):.2f}")
        return position_id

    def check_exits(self, current_prices: Dict):
        """CHECK IF POSITIONS SHOULD CLOSE"""
        for position_id, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue

            current_price = current_prices.get(position.symbol)
            if not current_price:
                continue

            # Calculate current PnL for logging
            if position.direction == "BUY":
                pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_percent = (position.entry_price - current_price) / position.entry_price * 100

            # Check target hit
            if position.direction == "BUY" and current_price >= position.target_price:
                self.close_position(position_id, current_price, f"TARGET_HIT (+{pnl_percent:.2f}%)")
            elif position.direction == "SELL" and current_price <= position.target_price:
                self.close_position(position_id, current_price, f"TARGET_HIT (+{pnl_percent:.2f}%)")

            # Check stop loss
            elif position.direction == "BUY" and current_price <= position.stop_loss:
                self.close_position(position_id, current_price, f"STOP_LOSS ({pnl_percent:.2f}%)")
            elif position.direction == "SELL" and current_price >= position.stop_loss:
                self.close_position(position_id, current_price, f"STOP_LOSS ({pnl_percent:.2f}%)")

    def close_position(self, position_id: str, exit_price: float, reason: str):
        """CLOSE POSITION AND UPDATE CAPITAL"""
        position = self.positions[position_id]

        if position.direction == "BUY":
            pnl_percent = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_percent = (position.entry_price - exit_price) / position.entry_price

        pnl_usd = position.size * pnl_percent

        position.status = "CLOSED"
        position.exit_price = exit_price
        position.pnl = pnl_usd
        self.capital += pnl_usd

        trade_log = {
            'timestamp': datetime.now(),
            'position_id': position_id,
            'strategy': position.strategy_id,
            'symbol': position.symbol,
            'action': 'CLOSE',
            'exit_price': exit_price,
            'pnl_usd': pnl_usd,
            'pnl_percent': pnl_percent * 100,
            'reason': reason
        }

        self.trade_history.append(trade_log)
        print(f"🔒 CLOSED: {position_id} | PnL: ${pnl_usd:+.2f} ({pnl_percent*100:+.2f}%) - {reason}")

# =============================================================================
# MAIN TRADING ENGINE - PROPER MANAGEMENT
# =============================================================================

class LiveTradingEngine:
    def __init__(self):
        self.htx_client = RealHTXClient()
        self.strategy_executor = StrategyExecutor()
        self.paper_engine = PaperTradingEngine()
        self.strategies = self.load_strategies()
        self.cycle_count = 0
        self.start_time = datetime.now()

    def load_strategies(self) -> Dict:
        strategies = {}

        print(f"🔍 Looking for strategies in: {Config.STRATEGIES_DIR}")

        if Config.STRATEGIES_DIR.exists():
            print(f"✅ Directory exists! Scanning for strategies...")

            py_files = [f for f in Config.STRATEGIES_DIR.glob("*.py") if '_meta.py' not in str(f)]

            print(f"📁 Found {len(py_files)} Python strategy files")

            for py_file in py_files:
                strategy_id = py_file.stem
                meta_filename = f"{strategy_id}_meta.json"
                meta_file = Config.STRATEGIES_DIR / meta_filename

                print(f"🔍 Strategy: {strategy_id}")

                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)

                        strategies[strategy_id] = {
                            'meta': meta_data,
                            'py_file': py_file,
                            'type': self.detect_strategy_type(meta_data.get('strategy_name', ''))
                        }
                        print(f"✅ LOADED: {strategy_id}")
                    except Exception as e:
                        print(f"❌ Failed to load {strategy_id}: {e}")
                else:
                    strategies[strategy_id] = {
                        'meta': {'strategy_name': strategy_id},
                        'py_file': py_file,
                        'type': self.detect_strategy_type(strategy_id)
                    }
                    print(f"✅ LOADED (NO META): {strategy_id}")
        else:
            print(f"❌ Strategies directory not found: {Config.STRATEGIES_DIR}")

        print(f"🎯 TOTAL STRATEGIES LOADED: {len(strategies)}")
        return strategies

    def detect_strategy_type(self, strategy_name: str) -> str:
        name_lower = str(strategy_name).lower()
        if 'vwap' in name_lower or 'mean' in name_lower:
            return 'vwap_mean_reversion'
        elif 'market' in name_lower and 'maker' in name_lower:
            return 'market_maker'
        elif 'pairs' in name_lower or 'cointegration' in name_lower:
            return 'pairs_trading'
        elif 'neural' in name_lower or 'ai' in name_lower:
            return 'ai_neural'
        elif 'stoikov' in name_lower:
            return 'market_maker'
        elif 'correlation' in name_lower:
            return 'pairs_trading'
        else:
            return 'vwap_mean_reversion'

    def execute_strategy_for_token(self, strategy_id: str, token: str):
        """EXECUTE STRATEGY WITH PROPER POSITION MANAGEMENT"""
        try:
            df = self.htx_client.fetch_candles(token, '15min', 100)
            if df is None or len(df) < 50:
                return

            current_price = self.htx_client.get_current_price(token)
            if not current_price:
                return

            strategy_type = self.strategies[strategy_id]['type']

            if strategy_type == 'vwap_mean_reversion':
                signal = self.strategy_executor.execute_vwap_strategy(df, current_price)
            else:
                signal = self.strategy_executor.execute_vwap_strategy(df, current_price)

            if signal['signal'] != 'HOLD':
                print(f"🚀 SIGNAL: {strategy_id} - {signal['signal']} {token}")
                print(f"   Reason: {signal['reason']}")
                self.paper_engine.open_position(strategy_id, token, signal)
            else:
                print(f"⏸️  HOLD: {strategy_id} {token} - {signal['reason']}")

        except Exception as e:
            print(f"❌ Error executing {strategy_id} for {token}: {e}")

    def display_status(self):
        open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
        closed_positions = [p for p in self.paper_engine.positions.values() if p.status == "CLOSED"]

        total_pnl = sum(p.pnl for p in closed_positions)

        print(f"\n{'='*80}")
        print(f"🔄 TRADEPEX LIVE - PROPER POSITION MANAGEMENT")
        print(f"{'='*80}")
        print(f"⏰ Cycle: {self.cycle_count} | Runtime: {datetime.now() - self.start_time}")
        print(f"💰 Capital: ${self.paper_engine.capital:.2f} | Total PnL: ${total_pnl:+.2f}")
        print(f"📊 Open: {len(open_positions)}/{Config.MAX_TOTAL_POSITIONS} | Closed: {len(closed_positions)}")
        print(f"🎯 Strategies: {len(self.strategies)} | Tokens: {len(Config.TRADEABLE_TOKENS)}")

        if open_positions:
            print(f"\n📊 OPEN POSITIONS:")
            for position in open_positions:
                current_price = self.htx_client.get_current_price(position.symbol) or position.entry_price
                if position.direction == "BUY":
                    pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    pnl_percent = (position.entry_price - current_price) / position.entry_price * 100

                print(f"   {position.strategy_id[:20]:<20} {position.symbol:<6} {position.direction:<4} "
                      f"Entry: ${position.entry_price:<8.2f} Current: ${current_price:<8.2f} "
                      f"Target: ${position.target_price:<8.2f} PnL: {pnl_percent:+.2f}%")
        else:
            print(f"\n📊 No open positions - monitoring for opportunities...")

        print(f"{'='*80}\n")

    def run_24_7(self):
        print("🚀 STARTING TRADEPEX 24/7 PAPER TRADING WITH PROPER MANAGEMENT...")
        print(f"🎯 Strategies: {len(self.strategies)}")
        print(f"💰 Tokens: {Config.TRADEABLE_TOKENS}")
        print(f"⏰ Check interval: {Config.CHECK_INTERVAL}s")
        print(f"📊 Position Limits: {Config.MAX_TOTAL_POSITIONS} total, {Config.MAX_POSITIONS_PER_STRATEGY} per strategy, {Config.MAX_POSITIONS_PER_TOKEN} per token")
        print("=" * 80)

        while True:
            self.cycle_count += 1

            try:
                print(f"\n🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

                current_prices = {}
                for token in Config.TRADEABLE_TOKENS:
                    price = self.htx_client.get_current_price(token)
                    if price:
                        current_prices[token] = price
                        print(f"   {token}: ${price:.2f}")

                # FIRST: Check for exits (close positions that hit targets/stops)
                self.paper_engine.check_exits(current_prices)

                # THEN: Look for new opportunities (if we have capacity)
                open_positions = [p for p in self.paper_engine.positions.values() if p.status == "OPEN"]
                if len(open_positions) < Config.MAX_TOTAL_POSITIONS:
                    for strategy_id in self.strategies:
                        for token in Config.TRADEABLE_TOKENS:
                            self.execute_strategy_for_token(strategy_id, token)
                            time.sleep(0.5)  # Rate limiting
                else:
                    print(f"⏸️  Position limit reached ({len(open_positions)}/{Config.MAX_TOTAL_POSITIONS}) - waiting for closes")

                self.display_status()
                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 SHUTDOWN REQUESTED...")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)

# =============================================================================
# START TRADING
# =============================================================================

if __name__ == "__main__":
    print("🎯 TRADEPEX 24/7 PAPER TRADING - PROPER POSITION MANAGEMENT")
    print("🔧 Initializing engines...")

    engine = LiveTradingEngine()
    engine.run_24_7()
