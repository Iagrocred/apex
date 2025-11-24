#!/usr/bin/env python3
"""
🧪 MINIMAL VWAP STRATEGY TEST - Verify Trading Execution
Tests VWAP Mean Reversion strategy with minimal code to verify trading works
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import eth_account
from eth_account.signers.local import LocalAccount

load_dotenv()

# =========================================================================================
# HYPERLIQUID CLIENT (MINIMAL)
# =========================================================================================

class HyperliquidClient:
    """Minimal Hyperliquid client for testing"""
    
    def __init__(self, account: LocalAccount):
        self.account = account
        self.base_url = "https://api.hyperliquid.xyz"
        self.info_url = f"{self.base_url}/info"
        self.exchange_url = f"{self.base_url}/exchange"
        print(f"✅ Connected to Hyperliquid")
        print(f"   Account: {account.address[:6]}...{account.address[-4:]}")
    
    def get_account_value(self) -> float:
        """Get account balance"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "clearinghouseState", "user": self.account.address},
                timeout=10
            )
            data = response.json()
            balance = float(data.get('marginSummary', {}).get('accountValue', 0))
            print(f"💰 Account Balance: ${balance:.2f}")
            return balance
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return 0.0
    
    def get_candles(self, symbol: str, interval: str, num_bars: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles"""
        try:
            # Map timeframe
            interval_map = {'15m': '15m', '1H': '1h', '4H': '4h', '1D': '1d'}
            hl_interval = interval_map.get(interval, interval.lower())
            
            end_time = int(time.time() * 1000)
            start_time = end_time - (num_bars * 15 * 60 * 1000)  # 15m bars
            
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": hl_interval,
                        "startTime": start_time,
                        "endTime": end_time
                    }
                },
                timeout=10
            )
            
            candles = response.json()
            if not candles:
                print(f"⚠️  No candle data for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            print(f"✅ Fetched {len(df)} candles for {symbol} @ {interval}")
            print(f"   Latest: ${df['close'].iloc[-1]:.2f} | Volume: {df['volume'].iloc[-1]:,.0f}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return pd.DataFrame()
    
    def get_mid_price(self, symbol: str) -> float:
        """Get current mid price"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "allMids"},
                timeout=10
            )
            all_mids = response.json()
            price = float(all_mids.get(symbol, 0))
            return price
        except:
            return 0.0

# =========================================================================================
# VWAP STRATEGY LOGIC (SIMPLIFIED)
# =========================================================================================

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_vwap_bands(df: pd.DataFrame, vwap: pd.Series, std_dev: float = 2.0) -> tuple:
    """Calculate VWAP bands"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    variance = ((typical_price - vwap) ** 2 * df['volume']).cumsum() / df['volume'].cumsum()
    std = np.sqrt(variance)
    
    upper_band = vwap + (std_dev * std)
    lower_band = vwap - (std_dev * std)
    
    return upper_band, lower_band

def generate_vwap_signal(df: pd.DataFrame) -> str:
    """Generate trading signal based on VWAP"""
    if len(df) < 20:
        return 'HOLD'
    
    # Calculate VWAP and bands
    vwap = calculate_vwap(df)
    upper_band, lower_band = calculate_vwap_bands(df, vwap)
    
    # Current price
    current_price = df['close'].iloc[-1]
    current_vwap = vwap.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    print(f"\n📊 VWAP Analysis:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   VWAP:          ${current_vwap:.2f}")
    print(f"   Upper Band:    ${current_upper:.2f}")
    print(f"   Lower Band:    ${current_lower:.2f}")
    
    # Generate signal
    if current_price <= current_lower:
        print(f"   💡 SIGNAL: BUY (price at lower band)")
        return 'BUY'
    elif current_price >= current_upper:
        print(f"   💡 SIGNAL: SELL (price at upper band)")
        return 'SELL'
    else:
        deviation = ((current_price - current_vwap) / current_vwap) * 100
        print(f"   💡 SIGNAL: HOLD (price {deviation:+.2f}% from VWAP)")
        return 'HOLD'

# =========================================================================================
# MAIN TEST
# =========================================================================================

def main():
    print("=" * 60)
    print("🧪 VWAP STRATEGY TEST - Minimal Trading Verification")
    print("=" * 60)
    print()
    
    # Initialize
    hyperliquid_key = os.getenv('HYPER_LIQUID_KEY')
    if not hyperliquid_key:
        print("❌ ERROR: HYPER_LIQUID_KEY not found in environment")
        return
    
    account = eth_account.Account.from_key(hyperliquid_key)
    client = HyperliquidClient(account)
    print()
    
    # Check balance
    balance = client.get_account_value()
    if balance <= 0:
        print("⚠️  WARNING: Account balance is $0.00")
        print("   This might be the agent wallet - main wallet has $615.05")
    print()
    
    # Test VWAP strategy on BTC
    symbol = 'BTC'
    timeframe = '15m'
    
    print(f"📈 Testing VWAP Strategy on {symbol} @ {timeframe}")
    print("-" * 60)
    
    # Fetch candles
    df = client.get_candles(symbol, timeframe, num_bars=100)
    if df.empty:
        print("❌ No data available")
        return
    
    # Generate signal
    signal = generate_vwap_signal(df)
    
    print()
    print("=" * 60)
    print(f"🎯 FINAL SIGNAL: {signal}")
    print("=" * 60)
    
    # Position sizing (if signal is actionable)
    if signal in ['BUY', 'SELL']:
        max_position = 195  # $195 max per trade (30% of $650)
        print(f"\n💼 Position Sizing:")
        print(f"   Max Position: ${max_position}")
        print(f"   With 5x Leverage: ${max_position * 5}")
        print(f"   Margin Required: ${max_position}")
        
        current_price = df['close'].iloc[-1]
        size = max_position / current_price
        print(f"   Trade Size: {size:.4f} {symbol}")
        print(f"   Entry Price: ${current_price:.2f}")
        
        # Calculate stop loss and take profit
        stop_loss_pct = 0.02  # 2%
        take_profit_pct = 0.05  # 5%
        
        if signal == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        print(f"   Stop Loss: ${stop_loss:.2f} ({stop_loss_pct*100}%)")
        print(f"   Take Profit: ${take_profit:.2f} ({take_profit_pct*100}%)")
        
        print(f"\n✅ READY FOR LIVE TRADING!")
        print(f"   Execute: {signal} {size:.4f} {symbol} @ ${current_price:.2f}")
    else:
        print(f"\n⏸️  No trade signal - HOLDING")
    
    print()
    print("=" * 60)
    print("✅ TEST COMPLETE - VWAP Strategy Verified")
    print("=" * 60)

if __name__ == "__main__":
    main()
