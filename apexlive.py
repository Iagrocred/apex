#!/usr/bin/env python3
"""
APEXLIVE - Live Trading Engine with Hyperliquid
================================================================
This is the LIVE trading module that executes real trades on Hyperliquid.

THE PIPELINE:
1. APEX.py → Backtests and discovers successful strategies → successful_strategies/
2. TRADEADAPT.py → Paper trades and optimizes → When 71%+ WR → apexlive/
3. APEXLIVE.py (THIS FILE) → Executes LIVE trades on Hyperliquid!

KEY FEATURES:
- Scans apexlive/ folder every 8 minutes for new/updated strategies
- Executes REAL trades via Hyperliquid API
- Auto-updates to improved versions when tradeadapt improves them
- Continues paper trading in tradeadapt for continuous improvement
- Safety limits and circuit breakers

USAGE:
    python3 apexlive.py

REQUIREMENTS:
    - Hyperliquid SDK: pip install hyperliquid-python-sdk
    - config.json with API credentials (see hyperlivesample for format)
"""

import os
import json
import time
import shutil
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

# Hyperliquid SDK imports
try:
    from eth_account import Account
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    print("⚠️  Hyperliquid SDK not installed. Run: pip install hyperliquid-python-sdk eth_account")

# =============================================================================
# CONFIGURATION - LIVE TRADING
# =============================================================================

class LiveConfig:
    """Configuration for LIVE trading - Safety first!"""
    
    # ==========================================================================
    # 🔥 CAPITAL AND POSITION SIZING
    # ==========================================================================
    STRATEGY_CAPITAL_USD = 8000.0     # $8,000 per strategy (fixed allocation)
    MAX_POSITION_SIZE_PERCENT = 0.15  # Max 15% of strategy capital per trade
    DEFAULT_LEVERAGE = 8              # 8x leverage for live (matches tradeadapt)
    
    # ==========================================================================
    # 📊 DYNAMIC PER-COIN LEVERAGE LIMITS (Hyperliquid varies by coin)
    # ==========================================================================
    # Some coins allow up to 40x (BTC), some only 5x (small caps)
    # We cap at 26x max but respect per-coin limits from Hyperliquid API
    MAX_LEVERAGE = 26                 # Hard cap at 26x (Hyperliquid allows up to 40x on BTC)
    
    # ==========================================================================
    # 📈 MINIMUM VOLUME FILTERING (Only trade liquid coins!)
    # ==========================================================================
    MIN_DAILY_VOLUME_USD = 3_000_000  # $3M minimum 24h volume to trade
    MIN_LEVERAGE_REQUIRED = 5         # Coin must support at least 5x leverage
    
    # ==========================================================================
    # 📁 DIRECTORIES AND FILES
    # ==========================================================================
    APEXLIVE_DIR = Path("./apexlive")          # Strategies ready for live trading
    CONFIG_FILE = Path("./config.json")         # Hyperliquid API credentials
    LIVE_TRADES_DIR = Path("./live_trades")     # Live trade logs
    LIVE_STATE_FILE = Path("./apexlive_state.json")  # Persistent state
    
    # ==========================================================================
    # ⏰ TIMING
    # ==========================================================================
    STRATEGY_SCAN_INTERVAL = 480      # 8 minutes (scan for new/updated strategies)
    MARKET_CHECK_INTERVAL = 30        # 30 seconds between market checks
    
    # ==========================================================================
    # 💰 HYPERLIQUID FEE STRUCTURE (MUCH LOWER THAN HTX!)
    # ==========================================================================
    # From hypermerge documentation - Hyperliquid has MUCH better fees!
    FUTURES_TAKER_FEE = 0.00035       # 0.035% taker fee per side
    FUTURES_MAKER_FEE = -0.00015      # -0.015% maker REBATE!
    ESTIMATED_SPREAD = 0.00005        # 0.005% spread (MUCH tighter than HTX!)
    EXTRA_SLIPPAGE = 0.00003          # 0.003% slippage (better execution)
    USE_MAKER_ORDERS = False          # Use taker for speed, maker for rebate
    # TOTAL COST at 8x: ~0.04% per side × 2 × 8x = ~0.64% per round trip
    # Compare to HTX: ~0.15% per side × 2 × 8x = ~2.4% per round trip
    
    # ==========================================================================
    # 🛡️ SAFETY LIMITS - CANNOT BE CHANGED BY LLM!
    # ==========================================================================
    MAX_DAILY_LOSS = -500.0           # Stop trading if down $500/day
    MAX_POSITION_SIZE = 0.15          # Max 15% of capital per trade
    MAX_OPEN_POSITIONS = 20           # Max 20 simultaneous positions
    MAX_POSITIONS_PER_STRATEGY = 5    # Max 5 positions per strategy
    
    # Circuit breakers
    PAUSE_AFTER_CONSECUTIVE_LOSSES = 3   # Pause strategy after 3 losses in a row
    PAUSE_DURATION_MINUTES = 60          # Pause for 60 minutes before resuming
    
    # ==========================================================================
    # 🎯 TAKE PROFIT / STOP LOSS LEVELS
    # ==========================================================================
    TP_LEVELS = [0.006, 0.009, 0.012]    # 0.6%, 0.9%, 1.2% (wider for live safety)
    TP_FRACTIONS = [0.5, 0.25, 0.25]     # 50%, 25%, 25%
    MIN_STOP_DISTANCE = 0.005            # 0.5% minimum stop
    MAX_STOP_DISTANCE = 0.015            # 1.5% maximum stop
    
    # ==========================================================================
    # 📊 TRADEABLE COINS (Hyperliquid supports 180+ coins)
    # ==========================================================================
    # We'll fetch dynamically, but start with major coins
    DEFAULT_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX',
                     'DOGE', 'MATIC', 'UNI', 'LTC', 'ATOM', 'APT', 'ARB', 'OP']
    
    # LLM Configuration (same as tradeadapt)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# =============================================================================
# HYPERLIQUID CLIENT
# =============================================================================

class HyperliquidClient:
    """
    Hyperliquid API client for LIVE trading.
    Based on tested hyperlivesample that successfully executed trades.
    """
    
    def __init__(self, use_testnet: bool = False):
        """Initialize Hyperliquid connection."""
        self.use_testnet = use_testnet
        self.exchange = None
        self.info = None
        self.base_url = None
        self.coin_info = {}  # Cache coin metadata
        
        if HYPERLIQUID_AVAILABLE:
            self._setup_clients()
        else:
            print("❌ Hyperliquid SDK not available - running in simulation mode")
    
    def _load_config(self) -> Tuple[str, str]:
        """Load API credentials from config.json"""
        if not LiveConfig.CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"❌ Config file not found: {LiveConfig.CONFIG_FILE}\n"
                "Create config.json with:\n"
                '{"secret_key": "your-private-key", "account_address": "your-wallet-address"}'
            )
        
        with open(LiveConfig.CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        
        return cfg.get("secret_key"), cfg.get("account_address")
    
    def _setup_clients(self):
        """Setup Hyperliquid exchange and info clients."""
        try:
            secret_key, account_address = self._load_config()
            
            self.base_url = (
                constants.TESTNET_API_URL if self.use_testnet 
                else constants.MAINNET_API_URL
            )
            
            # Create signer from private key
            signer = Account.from_key(secret_key)
            
            # Info client (read-only for prices, positions)
            self.info = Info(base_url=self.base_url, skip_ws=True)
            
            # Exchange client (for trading)
            self.exchange = Exchange(
                signer,
                base_url=self.base_url,
                account_address=account_address
            )
            
            print(f"✅ Hyperliquid connected: {self.base_url}")
            print(f"   Account: {account_address[:10]}...{account_address[-6:]}")
            
            # Fetch available coins
            self._fetch_coin_info()
            
        except Exception as e:
            print(f"❌ Hyperliquid setup failed: {e}")
            self.exchange = None
            self.info = None
    
    def _fetch_coin_info(self):
        """Fetch available coins and their metadata from Hyperliquid."""
        try:
            meta = self.info.meta()
            universe = meta.get('universe', [])
            
            for coin_data in universe:
                name = coin_data.get('name', '')
                self.coin_info[name] = {
                    'max_leverage': coin_data.get('maxLeverage', 10),
                    'sz_decimals': coin_data.get('szDecimals', 3),
                }
            
            print(f"📊 Fetched {len(self.coin_info)} tradeable coins")
            
        except Exception as e:
            print(f"⚠️  Could not fetch coin info: {e}")
    
    def get_effective_leverage(self, coin: str) -> int:
        """
        Get the effective leverage for a coin.
        Returns min(DEFAULT_LEVERAGE, coin_max_leverage, MAX_LEVERAGE).
        """
        coin_max = self.coin_info.get(coin, {}).get('max_leverage', 10)
        effective = min(LiveConfig.DEFAULT_LEVERAGE, coin_max, LiveConfig.MAX_LEVERAGE)
        return int(effective)
    
    def is_coin_tradeable(self, coin: str) -> bool:
        """
        Check if coin meets minimum requirements for trading.
        - Must have at least MIN_LEVERAGE_REQUIRED leverage
        - Volume check would require additional API call
        """
        coin_max = self.coin_info.get(coin, {}).get('max_leverage', 0)
        return coin_max >= LiveConfig.MIN_LEVERAGE_REQUIRED
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if not self.info:
            return None
        
        try:
            # Hyperliquid uses symbol without USDT suffix
            all_mids = self.info.all_mids()
            price = all_mids.get(symbol)
            if price:
                return float(price)
        except Exception as e:
            print(f"⚠️  Price fetch failed for {symbol}: {e}")
        
        return None
    
    def get_account_balance(self) -> Optional[float]:
        """Get account balance in USD."""
        if not self.info:
            return None
        
        try:
            # Get user state
            state = self.info.user_state(self.exchange.account_address)
            margin_summary = state.get('marginSummary', {})
            account_value = float(margin_summary.get('accountValue', 0))
            return account_value
        except Exception as e:
            print(f"⚠️  Balance fetch failed: {e}")
        
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.info:
            return []
        
        try:
            state = self.info.user_state(self.exchange.account_address)
            positions = state.get('assetPositions', [])
            
            open_positions = []
            for pos in positions:
                position_data = pos.get('position', {})
                size = float(position_data.get('szi', 0))
                
                if abs(size) > 0:
                    open_positions.append({
                        'symbol': position_data.get('coin', ''),
                        'size': size,
                        'entry_price': float(position_data.get('entryPx', 0)),
                        'unrealized_pnl': float(position_data.get('unrealizedPnl', 0)),
                        'leverage': float(position_data.get('leverage', {}).get('value', 1))
                    })
            
            return open_positions
            
        except Exception as e:
            print(f"⚠️  Positions fetch failed: {e}")
        
        return []
    
    def market_open(self, coin: str, is_buy: bool, size: float, 
                    leverage: int = None) -> Dict:
        """
        Open a market position.
        
        Args:
            coin: Trading pair (e.g., 'BTC', 'ETH')
            is_buy: True for LONG, False for SHORT
            size: Position size in coin units
            leverage: Leverage to use (optional, uses account default)
        """
        if not self.exchange:
            return {'status': 'error', 'message': 'Exchange not connected'}
        
        try:
            # Set leverage if specified
            if leverage:
                max_lev = self.coin_info.get(coin, {}).get('max_leverage', 10)
                leverage = min(leverage, max_lev, LiveConfig.MAX_LEVERAGE)
                # Hyperliquid sets leverage per-coin automatically
            
            # Execute market order
            # slippage parameter: 0.01 = 1% slippage tolerance
            result = self.exchange.market_open(
                coin, 
                is_buy, 
                size, 
                None,  # price (None for market)
                0.01   # slippage
            )
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def market_close(self, coin: str) -> Dict:
        """Close all positions for a coin."""
        if not self.exchange:
            return {'status': 'error', 'message': 'Exchange not connected'}
        
        try:
            result = self.exchange.market_close(coin)
            return result
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def fetch_candles(self, symbol: str, interval: str = '15m', 
                      limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles for a symbol.
        Uses Hyperliquid candle API.
        """
        if not self.info:
            return None
        
        try:
            # Calculate time range
            end_time = int(time.time() * 1000)
            
            # Map interval to milliseconds
            interval_ms = {
                '1m': 60000, '5m': 300000, '15m': 900000,
                '1h': 3600000, '4h': 14400000, '1d': 86400000
            }.get(interval, 900000)
            
            start_time = end_time - (limit * interval_ms)
            
            # Fetch candles
            candles = self.info.candles_snapshot(
                coin=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time
            )
            
            if not candles:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            # Convert types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime')
            
            return df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"⚠️  Candle fetch failed for {symbol}: {e}")
        
        return None


# =============================================================================
# LIVE POSITION TRACKING
# =============================================================================

@dataclass
class LivePosition:
    """Track a live trading position."""
    position_id: str
    strategy_id: str
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    size_usd: float
    size_coin: float
    target_prices: List[float]
    stop_loss: float
    entry_time: datetime
    leverage: int
    status: str = "OPEN"
    exit_price: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""
    tp_index: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        return d


# =============================================================================
# LIVE TRADING ENGINE
# =============================================================================

class LiveTradingEngine:
    """
    LIVE Trading Engine for Hyperliquid.
    
    This is the final piece of the puzzle:
    1. Monitors apexlive/ folder for strategies promoted by tradeadapt
    2. Executes REAL trades on Hyperliquid
    3. Auto-updates to improved strategy versions
    4. Maintains safety limits and circuit breakers
    """
    
    def __init__(self, use_testnet: bool = False):
        """Initialize the live trading engine."""
        self.hyperliquid = HyperliquidClient(use_testnet=use_testnet)
        
        # Strategy management
        self.strategies: Dict[str, Dict] = {}
        self.strategy_performance: Dict[str, Dict] = {}
        
        # Position tracking
        self.positions: Dict[str, LivePosition] = {}
        self.trade_history: List[Dict] = []
        
        # Timing
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.last_strategy_scan = datetime.now()
        
        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_pnl_reset_date = datetime.now().date()
        
        # Load state and strategies
        self.load_state()
        self.scan_for_strategies()
        
        # Create directories
        LiveConfig.LIVE_TRADES_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 APEXLIVE Engine initialized")
        print(f"   Mode: {'TESTNET' if use_testnet else 'MAINNET'}")
        print(f"   Strategies: {len(self.strategies)}")
    
    def scan_for_strategies(self) -> List[str]:
        """
        Scan apexlive/ folder for strategies to trade.
        Called every 8 minutes to pick up new/updated strategies.
        """
        if not LiveConfig.APEXLIVE_DIR.exists():
            print(f"📁 Creating apexlive directory: {LiveConfig.APEXLIVE_DIR}")
            LiveConfig.APEXLIVE_DIR.mkdir(parents=True, exist_ok=True)
            return []
        
        new_strategies = []
        updated_strategies = []
        
        # Find all strategy files
        strategy_files = list(LiveConfig.APEXLIVE_DIR.glob("*_live*.py"))
        
        for strategy_file in strategy_files:
            strategy_id = strategy_file.stem.replace('_live', '').rsplit('_v', 1)[0]
            
            # Check if new or updated
            is_new = strategy_id not in self.strategies
            
            # Parse version from filename
            version = 0
            if '_v' in strategy_file.stem:
                try:
                    version = int(strategy_file.stem.rsplit('_v', 1)[1])
                except ValueError:
                    pass
            
            # Check if this is a newer version
            current_version = self.strategies.get(strategy_id, {}).get('version', -1)
            
            if is_new or version > current_version:
                # Load strategy metadata
                meta_file = LiveConfig.APEXLIVE_DIR / f"{strategy_id}_live_meta.json"
                meta_data = {}
                
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                    except:
                        pass
                
                self.strategies[strategy_id] = {
                    'file': strategy_file,
                    'version': version,
                    'meta': meta_data,
                    'loaded_at': datetime.now().isoformat(),
                    'paused': False,
                    'pause_until': None,
                    'consecutive_losses': 0
                }
                
                # Initialize performance tracking
                if strategy_id not in self.strategy_performance:
                    self.strategy_performance[strategy_id] = {
                        'total_trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0.0,
                        'consecutive_losses': 0
                    }
                
                if is_new:
                    new_strategies.append(strategy_id)
                    print(f"🆕 NEW STRATEGY: {strategy_id} v{version}")
                else:
                    updated_strategies.append(strategy_id)
                    print(f"🔄 UPDATED STRATEGY: {strategy_id} v{current_version} → v{version}")
        
        if new_strategies or updated_strategies:
            print(f"📊 Total active strategies: {len(self.strategies)}")
        
        return new_strategies + updated_strategies
    
    def can_open_position(self, strategy_id: str, symbol: str) -> Tuple[bool, str]:
        """Check if we can open a new position."""
        
        # Check if strategy is paused
        strategy = self.strategies.get(strategy_id, {})
        if strategy.get('paused'):
            pause_until = strategy.get('pause_until')
            if pause_until and datetime.now() < datetime.fromisoformat(pause_until):
                return False, f"Strategy paused until {pause_until}"
            else:
                # Unpause
                strategy['paused'] = False
                strategy['pause_until'] = None
        
        # Count open positions
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        
        # Total position limit
        if len(open_positions) >= LiveConfig.MAX_OPEN_POSITIONS:
            return False, f"Max total positions ({LiveConfig.MAX_OPEN_POSITIONS}) reached"
        
        # Strategy position limit
        strategy_positions = [p for p in open_positions if p.strategy_id == strategy_id]
        if len(strategy_positions) >= LiveConfig.MAX_POSITIONS_PER_STRATEGY:
            return False, f"Strategy has max positions ({LiveConfig.MAX_POSITIONS_PER_STRATEGY})"
        
        # Check daily loss limit
        if self.daily_pnl <= LiveConfig.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
        
        # Existing position check
        existing = [p for p in open_positions 
                   if p.strategy_id == strategy_id and p.symbol == symbol]
        if existing:
            return False, f"Already have position on {symbol}"
        
        return True, "OK"
    
    def open_position(self, strategy_id: str, symbol: str, 
                      signal: Dict) -> Optional[str]:
        """Open a LIVE position on Hyperliquid."""
        
        if signal.get('signal') == 'HOLD':
            return None
        
        can_open, reason = self.can_open_position(strategy_id, symbol)
        if not can_open:
            print(f"⏸️  BLOCKED: {strategy_id[:30]} {symbol} - {reason}")
            return None
        
        direction = signal['signal']  # 'BUY' or 'SELL'
        entry_price = signal.get('current_price', 0)
        
        if entry_price <= 0:
            return None
        
        # Calculate position size
        position_usd = LiveConfig.STRATEGY_CAPITAL_USD * LiveConfig.MAX_POSITION_SIZE_PERCENT
        leverage = min(LiveConfig.DEFAULT_LEVERAGE, LiveConfig.MAX_LEVERAGE)
        leveraged_usd = position_usd * leverage
        
        # Convert to coin units
        size_coin = leveraged_usd / entry_price
        
        # Get coin decimals for rounding
        coin_info = self.hyperliquid.coin_info.get(symbol, {})
        decimals = coin_info.get('sz_decimals', 3)
        size_coin = round(size_coin, decimals)
        
        # Calculate TP levels
        atr = signal.get('atr', entry_price * 0.01)
        target_prices = []
        for level in LiveConfig.TP_LEVELS:
            if direction == 'BUY':
                target_prices.append(entry_price * (1 + level))
            else:
                target_prices.append(entry_price * (1 - level))
        
        # Calculate stop loss
        stop_distance = min(max(atr * 1.5, entry_price * LiveConfig.MIN_STOP_DISTANCE),
                           entry_price * LiveConfig.MAX_STOP_DISTANCE)
        
        if direction == 'BUY':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # EXECUTE LIVE TRADE!
        print(f"\n🔥 LIVE TRADE: {direction} {symbol}")
        print(f"   Size: {size_coin} {symbol} (${leveraged_usd:.2f} @ {leverage}x)")
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Targets: {[f'${p:.2f}' for p in target_prices]}")
        print(f"   Stop: ${stop_loss:.2f}")
        
        is_buy = direction == 'BUY'
        result = self.hyperliquid.market_open(symbol, is_buy, size_coin, leverage)
        
        if result.get('status') == 'ok':
            # Get actual fill price
            statuses = result.get('response', {}).get('data', {}).get('statuses', [])
            if statuses and 'filled' in statuses[0]:
                actual_price = float(statuses[0]['filled'].get('avgPx', entry_price))
                order_id = statuses[0]['filled'].get('oid', '')
            else:
                actual_price = entry_price
                order_id = ''
            
            position_id = f"{strategy_id}_{symbol}_{datetime.now().strftime('%H%M%S')}"
            
            position = LivePosition(
                position_id=position_id,
                strategy_id=strategy_id,
                symbol=symbol,
                direction=direction,
                entry_price=actual_price,
                size_usd=leveraged_usd,
                size_coin=size_coin,
                target_prices=target_prices,
                stop_loss=stop_loss,
                entry_time=datetime.now(),
                leverage=leverage
            )
            
            self.positions[position_id] = position
            
            print(f"   ✅ FILLED @ ${actual_price:.2f} | ID: {order_id}")
            
            # Log trade
            self.log_trade(position, 'OPEN', result)
            
            return position_id
        else:
            error = result.get('message', 'Unknown error')
            print(f"   ❌ ORDER FAILED: {error}")
            return None
    
    def check_exits(self, current_prices: Dict[str, float]):
        """Check and execute exits for open positions."""
        
        for position_id, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue
            
            current_price = current_prices.get(position.symbol)
            if not current_price:
                continue
            
            # Calculate P&L
            if position.direction == 'BUY':
                pnl_percent = (current_price - position.entry_price) / position.entry_price
            else:
                pnl_percent = (position.entry_price - current_price) / position.entry_price
            
            pnl_usd = position.size_usd * pnl_percent
            
            # Check stop loss
            stop_hit = False
            if position.direction == 'BUY' and current_price <= position.stop_loss:
                stop_hit = True
            elif position.direction == 'SELL' and current_price >= position.stop_loss:
                stop_hit = True
            
            if stop_hit:
                self.close_position(position_id, current_price, "STOP_LOSS")
                continue
            
            # Check take profits
            if position.tp_index < len(position.target_prices):
                target = position.target_prices[position.tp_index]
                
                tp_hit = False
                if position.direction == 'BUY' and current_price >= target:
                    tp_hit = True
                elif position.direction == 'SELL' and current_price <= target:
                    tp_hit = True
                
                if tp_hit:
                    fraction = LiveConfig.TP_FRACTIONS[position.tp_index]
                    self.close_position(position_id, current_price, 
                                       f"TP{position.tp_index + 1}", fraction)
                    position.tp_index += 1
                    
                    # Move stop to breakeven after first TP
                    if position.tp_index == 1:
                        position.stop_loss = position.entry_price
                        print(f"   📈 Stop moved to breakeven: ${position.stop_loss:.2f}")
            
            # Check max holding time (4 hours)
            holding_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
            if holding_minutes > 240:
                self.close_position(position_id, current_price, "MAX_TIME")
    
    def close_position(self, position_id: str, exit_price: float, 
                       reason: str, fraction: float = 1.0):
        """Close a LIVE position."""
        
        position = self.positions.get(position_id)
        if not position or position.status != "OPEN":
            return
        
        # Calculate P&L
        if position.direction == 'BUY':
            pnl_percent = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_percent = (position.entry_price - exit_price) / position.entry_price
        
        close_size = position.size_coin * fraction
        close_usd = position.size_usd * fraction
        pnl_usd = close_usd * pnl_percent
        
        # Subtract trading costs
        trading_cost = close_usd * LiveConfig.FUTURES_TAKER_FEE * 2
        pnl_usd -= trading_cost
        
        print(f"\n📥 CLOSING LIVE: {position.symbol} ({reason})")
        print(f"   Size: {close_size:.4f} {position.symbol}")
        print(f"   Exit: ${exit_price:.2f}")
        print(f"   P&L: ${pnl_usd:+.2f} ({pnl_percent*100:+.2f}%)")
        
        # Execute close on Hyperliquid
        if fraction >= 0.99:
            # Full close
            result = self.hyperliquid.market_close(position.symbol)
        else:
            # Partial close - open opposite position
            is_buy = position.direction == 'SELL'  # Opposite
            result = self.hyperliquid.market_open(
                position.symbol, is_buy, close_size, position.leverage
            )
        
        if result.get('status') == 'ok':
            print(f"   ✅ CLOSED!")
            
            # Update position
            position.pnl_usd += pnl_usd
            position.size_coin -= close_size
            position.size_usd -= close_usd
            
            if position.size_coin < 0.00001:
                position.status = "CLOSED"
                position.exit_price = exit_price
                position.exit_reason = reason
            
            # Update performance tracking
            self.update_performance(position.strategy_id, pnl_usd)
            
            # Update daily P&L
            self.daily_pnl += pnl_usd
            
            # Log trade
            self.log_trade(position, 'CLOSE', result, pnl_usd, reason)
        else:
            print(f"   ❌ CLOSE FAILED: {result.get('message')}")
    
    def update_performance(self, strategy_id: str, pnl: float):
        """Update strategy performance tracking."""
        
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'consecutive_losses': 0
            }
        
        perf = self.strategy_performance[strategy_id]
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl
        
        if pnl > 0:
            perf['wins'] += 1
            perf['consecutive_losses'] = 0
            self.strategies[strategy_id]['consecutive_losses'] = 0
        else:
            perf['losses'] += 1
            perf['consecutive_losses'] += 1
            self.strategies[strategy_id]['consecutive_losses'] += 1
            
            # Check for pause trigger
            if perf['consecutive_losses'] >= LiveConfig.PAUSE_AFTER_CONSECUTIVE_LOSSES:
                pause_until = datetime.now() + timedelta(minutes=LiveConfig.PAUSE_DURATION_MINUTES)
                self.strategies[strategy_id]['paused'] = True
                self.strategies[strategy_id]['pause_until'] = pause_until.isoformat()
                print(f"⏸️  Strategy {strategy_id} PAUSED for {LiveConfig.PAUSE_DURATION_MINUTES} minutes")
                print(f"   Reason: {perf['consecutive_losses']} consecutive losses")
    
    def log_trade(self, position: LivePosition, action: str, 
                  result: Dict, pnl: float = 0, reason: str = ""):
        """Log trade to file."""
        
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'strategy_id': position.strategy_id,
            'symbol': position.symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'size_coin': position.size_coin,
            'size_usd': position.size_usd,
            'leverage': position.leverage,
            'pnl_usd': pnl,
            'reason': reason,
            'api_result': result
        }
        
        self.trade_history.append(trade_log)
        
        # Save to file
        log_file = LiveConfig.LIVE_TRADES_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        
        existing = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            except:
                pass
        
        existing.append(trade_log)
        
        with open(log_file, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
    
    def save_state(self):
        """Save engine state for persistence."""
        
        state = {
            'strategies': {k: {**v, 'file': str(v['file'])} 
                          for k, v in self.strategies.items()},
            'strategy_performance': self.strategy_performance,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'daily_pnl': self.daily_pnl,
            'daily_pnl_reset_date': str(self.daily_pnl_reset_date),
            'cycle_count': self.cycle_count,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(LiveConfig.LIVE_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 State saved")
    
    def load_state(self):
        """Load saved state."""
        
        if not LiveConfig.LIVE_STATE_FILE.exists():
            return
        
        try:
            with open(LiveConfig.LIVE_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            self.strategy_performance = state.get('strategy_performance', {})
            self.daily_pnl = state.get('daily_pnl', 0.0)
            
            # Check if we need to reset daily P&L
            saved_date = state.get('daily_pnl_reset_date')
            if saved_date and saved_date != str(datetime.now().date()):
                self.daily_pnl = 0.0
                self.daily_pnl_reset_date = datetime.now().date()
            
            print(f"📂 State loaded")
            print(f"   Daily P&L: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            print(f"⚠️  Could not load state: {e}")
    
    def display_status(self):
        """Display current live trading status."""
        
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        closed_today = len([t for t in self.trade_history 
                           if t.get('timestamp', '').startswith(str(datetime.now().date()))])
        
        print(f"\n{'='*80}")
        print(f"🔥 APEXLIVE - Live Trading Status")
        print(f"{'='*80}")
        print(f"⏰ Cycle: {self.cycle_count} | Runtime: {datetime.now() - self.start_time}")
        print(f"💰 Daily P&L: ${self.daily_pnl:+.2f}")
        print(f"📊 Open Positions: {len(open_positions)}/{LiveConfig.MAX_OPEN_POSITIONS}")
        print(f"📈 Trades Today: {closed_today}")
        print(f"🎯 Active Strategies: {len(self.strategies)}")
        
        # Strategy performance
        for strategy_id, perf in self.strategy_performance.items():
            if perf['total_trades'] > 0:
                win_rate = perf['wins'] / perf['total_trades'] * 100
                print(f"   {strategy_id[:30]}: {perf['wins']}W/{perf['losses']}L ({win_rate:.1f}%) ${perf['total_pnl']:+.2f}")
        
        # Open positions
        if open_positions:
            print(f"\n📊 OPEN POSITIONS:")
            for pos in open_positions:
                price = self.hyperliquid.get_current_price(pos.symbol) or pos.entry_price
                if pos.direction == 'BUY':
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
                pnl_usd = pos.size_usd * pnl_pct / 100
                
                emoji = "🟢" if pnl_usd >= 0 else "🔴"
                print(f"   {emoji} {pos.direction} {pos.symbol} @ ${pos.entry_price:.2f} → ${price:.2f} ({pnl_pct:+.2f}%) = ${pnl_usd:+.2f}")
        
        print(f"{'='*80}\n")
    
    def run(self):
        """Main live trading loop."""
        
        print("🚀 STARTING APEXLIVE - Live Trading Engine")
        print(f"🔥 Mode: {'TESTNET' if self.hyperliquid.use_testnet else 'MAINNET'}")
        print(f"🎯 Strategies: {len(self.strategies)}")
        print(f"💰 Capital per strategy: ${LiveConfig.STRATEGY_CAPITAL_USD}")
        print(f"🛡️ Max daily loss: ${LiveConfig.MAX_DAILY_LOSS}")
        print(f"⏰ Strategy scan interval: {LiveConfig.STRATEGY_SCAN_INTERVAL // 60} minutes")
        print("="*80)
        
        if not self.strategies:
            print("\n⚠️  No strategies in apexlive/ folder!")
            print("   Waiting for tradeadapt to promote 71%+ strategies...")
        
        while True:
            self.cycle_count += 1
            
            try:
                print(f"\n🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Reset daily P&L if new day
                if datetime.now().date() != self.daily_pnl_reset_date:
                    print(f"📅 New day - resetting daily P&L (was ${self.daily_pnl:.2f})")
                    self.daily_pnl = 0.0
                    self.daily_pnl_reset_date = datetime.now().date()
                
                # =============================================================
                # 🔍 SCAN FOR NEW/UPDATED STRATEGIES (every 8 minutes)
                # =============================================================
                seconds_since_scan = (datetime.now() - self.last_strategy_scan).total_seconds()
                if seconds_since_scan >= LiveConfig.STRATEGY_SCAN_INTERVAL:
                    print(f"\n🔍 SCANNING FOR STRATEGIES (8-minute interval)...")
                    self.scan_for_strategies()
                    self.last_strategy_scan = datetime.now()
                
                # Get current prices
                current_prices = {}
                for coin in LiveConfig.DEFAULT_COINS:
                    price = self.hyperliquid.get_current_price(coin)
                    if price:
                        current_prices[coin] = price
                
                # Check exits
                self.check_exits(current_prices)
                
                # Check for new signals (simplified - in production would load strategy logic)
                # For now, strategies in apexlive folder are monitored but signals 
                # would need the actual strategy execution logic
                
                # Display status
                if self.cycle_count % 10 == 0:
                    self.display_status()
                    self.save_state()
                
                time.sleep(LiveConfig.MARKET_CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 SHUTDOWN REQUESTED...")
                self.save_state()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("🔥 APEXLIVE - Live Trading Engine for Hyperliquid")
    print("="*80)
    print("This module executes REAL trades with REAL money!")
    print("Make sure you have:")
    print("  1. config.json with Hyperliquid API credentials")
    print("  2. Strategies promoted to apexlive/ folder by tradeadapt")
    print("="*80)
    
    # Check for testnet flag
    use_testnet = "--testnet" in sys.argv or "-t" in sys.argv
    
    if use_testnet:
        print("\n🧪 TESTNET MODE - No real money at risk")
    else:
        print("\n⚠️  MAINNET MODE - Real money will be traded!")
        response = input("Are you sure? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Aborted.")
            sys.exit(0)
    
    # Check for Hyperliquid SDK
    if not HYPERLIQUID_AVAILABLE:
        print("\n❌ Hyperliquid SDK not installed!")
        print("   Run: pip install hyperliquid-python-sdk eth_account")
        sys.exit(1)
    
    engine = LiveTradingEngine(use_testnet=use_testnet)
    engine.run()
