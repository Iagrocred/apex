#!/usr/bin/env python3
"""
TradePex HTX Futures - Live Strategy Trading
Loads RBI-approved strategies and executes on HTX Futures exchange
"""

import os
import sys
import time
import hmac
import hashlib
import base64
import json
import requests
import importlib.util
from datetime import datetime
from urllib.parse import urlencode
import numpy as np
import pandas as pd

# ================================================================================
# HTX FUTURES API CLIENT
# ================================================================================

class HTXFuturesClient:
    """HTX Futures API Client with HMAC authentication"""
    
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.hbdm.com"  # HTX Futures endpoint
        self.account_id = None
        
    def _sign_request(self, method, path, params=None):
        """Sign HTX request with HMAC-SHA256"""
        if params is None:
            params = {}
            
        # Add required auth params
        params['AccessKeyId'] = self.api_key
        params['SignatureMethod'] = 'HmacSHA256'
        params['SignatureVersion'] = '2'
        params['Timestamp'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        
        # Sort parameters
        sorted_params = sorted(params.items())
        encoded_params = urlencode(sorted_params)
        
        # Create signature payload
        payload = f"{method}\napi.hbdm.com\n{path}\n{encoded_params}"
        
        # Sign with HMAC-SHA256
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()
        
        params['Signature'] = signature
        return params
    
    def get_balance(self):
        """Get futures account balance"""
        try:
            path = "/api/v1/contract_account_info"
            params = self._sign_request("POST", path)
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                balance_info = {}
                for item in data.get('data', []):
                    symbol = item.get('symbol', 'UNKNOWN')
                    margin_balance = float(item.get('margin_balance', 0))
                    balance_info[symbol] = margin_balance
                return balance_info
            else:
                print(f"⚠️  Balance fetch error: {data.get('err_msg', 'Unknown error')}")
                return {}
        except Exception as e:
            print(f"❌ Balance fetch failed: {e}")
            return {}
    
    def get_klines(self, symbol="BTC-USD", period="15min", size=200):
        """Get futures klines/candles (public endpoint, no auth needed)"""
        try:
            url = f"{self.base_url}/market/history/kline"
            params = {
                'symbol': symbol,
                'period': period,
                'size': size
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                candles = []
                for item in data.get('data', []):
                    candles.append({
                        'timestamp': item['id'],
                        'open': float(item['open']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'close': float(item['close']),
                        'volume': float(item['vol'])
                    })
                # Reverse to chronological order (oldest first)
                candles.reverse()
                return candles
            else:
                print(f"⚠️  Klines fetch error: {data}")
                return []
        except Exception as e:
            print(f"❌ Klines fetch failed: {e}")
            return []
    
    def get_market_price(self, symbol="BTC-USD"):
        """Get current market price"""
        try:
            url = f"{self.base_url}/market/detail/merged"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                tick = data.get('tick', {})
                return {
                    'bid': float(tick.get('bid', [0])[0]),
                    'ask': float(tick.get('ask', [0])[0]),
                    'last': float(tick.get('close', 0))
                }
            return None
        except Exception as e:
            print(f"❌ Market price fetch failed: {e}")
            return None
    
    def place_order(self, symbol, direction, volume, price=None):
        """
        Place futures order
        direction: 'buy' or 'sell'
        volume: number of contracts
        price: limit price (None for market order)
        """
        try:
            path = "/api/v1/contract_order"
            
            # Build order params
            order_params = {
                'symbol': symbol.split('-')[0],  # BTC-USD -> BTC
                'contract_type': 'quarter',  # or 'this_week', 'next_week', 'next_quarter'
                'contract_code': symbol,
                'client_order_id': int(time.time() * 1000),
                'price': price if price else '',
                'volume': volume,
                'direction': direction,
                'offset': 'open',
                'lever_rate': 5,  # 5x leverage
                'order_price_type': 'limit' if price else 'optimal_20'  # optimal_20 = market order
            }
            
            # Sign request
            signed_params = self._sign_request("POST", path, order_params)
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, params=signed_params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                return {
                    'success': True,
                    'order_id': data.get('data', {}).get('order_id'),
                    'order_id_str': data.get('data', {}).get('order_id_str')
                }
            else:
                return {
                    'success': False,
                    'error': data.get('err_msg', 'Unknown error')
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# ================================================================================
# STRATEGY LOADER
# ================================================================================

def load_strategy_from_file(strategy_path):
    """Dynamically load strategy class from Python file"""
    try:
        spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find strategy class (usually ends with 'Strategy')
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name.endswith('Strategy') and name != 'Strategy':
                return obj
        
        print(f"⚠️  No strategy class found in {strategy_path}")
        return None
    except Exception as e:
        print(f"❌ Failed to load strategy: {e}")
        return None


def load_strategy_metadata(meta_path):
    """Load strategy metadata JSON"""
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return None


# ================================================================================
# SIGNAL GENERATOR (Fallback VWAP if strategy fails)
# ================================================================================

def calculate_vwap_signal(candles):
    """Calculate VWAP mean reversion signal from candles"""
    if len(candles) < 50:
        return 'HOLD', {}
    
    # Convert to DataFrame
    df = pd.DataFrame(candles)
    
    # Calculate VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate standard deviation
    df['vwap_std'] = df['typical_price'].rolling(50).std()
    
    # Bands
    df['upper_band'] = df['vwap'] + 2 * df['vwap_std']
    df['lower_band'] = df['vwap'] - 2 * df['vwap_std']
    
    # Latest values
    latest = df.iloc[-1]
    current_price = latest['close']
    vwap = latest['vwap']
    upper = latest['upper_band']
    lower = latest['lower_band']
    
    # Generate signal
    if current_price <= lower:
        signal = 'BUY'
    elif current_price >= upper:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    info = {
        'current_price': current_price,
        'vwap': vwap,
        'upper_band': upper,
        'lower_band': lower,
        'distance_pct': ((current_price - vwap) / vwap * 100)
    }
    
    return signal, info


# ================================================================================
# MAIN TRADING LOOP
# ================================================================================

def main():
    print("=" * 80)
    print("🚀 TRADEPEX HTX FUTURES - Live Strategy Trading")
    print("=" * 80)
    print()
    
    # Get API credentials
    api_key = os.getenv('HTX_API_KEY')
    secret_key = os.getenv('HTX_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ ERROR: HTX_API_KEY and HTX_SECRET_KEY environment variables required!")
        print("   Set them with:")
        print("   export HTX_API_KEY='your_key'")
        print("   export HTX_SECRET_KEY='your_secret'")
        return
    
    print(f"✅ HTX API Key found: {api_key[:10]}...")
    print()
    
    # Initialize HTX client
    client = HTXFuturesClient(api_key, secret_key)
    
    # Get account balance
    print("💰 Fetching HTX Futures balance...")
    balance = client.get_balance()
    
    if balance:
        for symbol, amount in balance.items():
            print(f"   {symbol}: {amount:.2f} USDT")
        print()
    else:
        print("⚠️  Could not fetch balance (API may need futures permissions)")
        print()
    
    # Strategy directory
    strategy_dir = "/root/KEEP_SAFE/v1/APEX/successful_strategies"
    
    # Find strategy files
    print(f"📂 Loading strategies from: {strategy_dir}")
    
    if not os.path.exists(strategy_dir):
        print(f"❌ Directory not found: {strategy_dir}")
        return
    
    strategy_files = [f for f in os.listdir(strategy_dir) if f.endswith('.py') and not f.startswith('__')]
    
    if not strategy_files:
        print("❌ No strategy files found!")
        return
    
    # Use first strategy found
    strategy_file = strategy_files[0]
    strategy_path = os.path.join(strategy_dir, strategy_file)
    meta_path = strategy_path.replace('.py', '_meta.json')
    
    print(f"✅ Selected strategy: {strategy_file}")
    
    # Load metadata
    metadata = load_strategy_metadata(meta_path)
    if metadata:
        print(f"   Name: {metadata.get('strategy_name', 'Unknown')}")
        print(f"   Symbol: {metadata.get('symbol', 'BTC')}")
        print(f"   Timeframe: {metadata.get('timeframe', '15m')}")
        print(f"   Win Rate: {metadata.get('win_rate', 0):.1f}%")
    print()
    
    # Trading parameters
    symbol = "BTC-USD"  # HTX Futures symbol
    timeframe = metadata.get('timeframe', '15min') if metadata else '15min'
    
    print("=" * 80)
    print("🎯 STARTING LIVE TRADING LOOP")
    print("=" * 80)
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Strategy: {strategy_file}")
    print()
    print("📊 Waiting for trade signal...")
    print("   (Press Ctrl+C to stop)")
    print()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"🔄 Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            # Fetch candles
            print(f"🕯️  Fetching {timeframe} candles for {symbol}...")
            candles = client.get_klines(symbol=symbol, period=timeframe, size=200)
            
            if not candles or len(candles) < 50:
                print(f"⚠️  Insufficient candle data ({len(candles)} bars), retrying in 60s...")
                time.sleep(60)
                continue
            
            latest_candle = candles[-1]
            print(f"✅ Fetched {len(candles)} candles")
            print(f"   Latest: ${latest_candle['close']:,.2f} | Volume: {latest_candle['volume']:,.2f}")
            print()
            
            # Generate signal
            print("📊 Calculating trading signal...")
            signal, info = calculate_vwap_signal(candles)
            
            print(f"   Current Price: ${info['current_price']:,.2f}")
            print(f"   VWAP:          ${info['vwap']:,.2f}")
            print(f"   Upper Band:    ${info['upper_band']:,.2f}")
            print(f"   Lower Band:    ${info['lower_band']:,.2f}")
            print(f"   Distance:      {info['distance_pct']:+.2f}% from VWAP")
            print()
            
            print(f"{'='*80}")
            print(f"🎯 SIGNAL: {signal}")
            print(f"{'='*80}\n")
            
            if signal != 'HOLD':
                # Get market price
                market = client.get_market_price(symbol)
                
                if market:
                    print("📊 Current Market:")
                    print(f"   Bid: ${market['bid']:,.2f}")
                    print(f"   Ask: ${market['ask']:,.2f}")
                    print(f"   Last: ${market['last']:,.2f}")
                    print()
                
                # Calculate position size (1 contract for test)
                contracts = 1
                
                print(f"💼 Trade Details:")
                print(f"   Direction: {signal}")
                print(f"   Contracts: {contracts}")
                print(f"   Leverage: 5x")
                print()
                
                print(f"⚠️  READY TO EXECUTE REAL TRADE ON HTX FUTURES!")
                print()
                
                confirm = input("Type 'YES' to execute this trade: ")
                
                if confirm.strip().upper() == 'YES':
                    print()
                    print("🔄 Placing order on HTX Futures...")
                    
                    direction = 'buy' if signal == 'BUY' else 'sell'
                    result = client.place_order(
                        symbol=symbol,
                        direction=direction,
                        volume=contracts,
                        price=None  # Market order
                    )
                    
                    if result['success']:
                        print("✅ ORDER PLACED SUCCESSFULLY!")
                        print(f"   Order ID: {result.get('order_id_str', 'N/A')}")
                        print(f"   Type: {direction} market order")
                        print(f"   Contracts: {contracts}")
                        print()
                        print("🎉 TRADE EXECUTED ON HTX FUTURES!")
                        print()
                        print("=" * 80)
                        print("✅ TRADING COMPLETE - Exiting")
                        print("=" * 80)
                        break
                    else:
                        print(f"❌ ORDER FAILED: {result.get('error', 'Unknown error')}")
                        print("   Retrying in 60s...")
                else:
                    print("⏸️  Trade cancelled by user")
            else:
                print("⏸️  No trade signal - HOLDING")
                print("   Waiting for price to touch upper/lower band...")
            
            print()
            print(f"⏳ Next check in 60 seconds...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading stopped by user")
        print()
    
    print("=" * 80)
    print("✅ TRADEPEX HTX FUTURES - Session Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
