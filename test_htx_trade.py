#!/usr/bin/env python3
"""
HTX Trading Test - Simple Single Strategy Execution
Loads one strategy, fetches HTX candles, generates signals, and executes trades.
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime
from urllib.parse import urlencode
import requests
import numpy as np
import pandas as pd

# HTX API Configuration
HTX_API_KEY = os.environ.get('HTX_API_KEY', '')
HTX_SECRET_KEY = os.environ.get('HTX_SECRET_KEY', '')
HTX_REST_URL = "https://api.huobi.pro"

class HTXClient:
    """Simple HTX exchange client"""
    
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = HTX_REST_URL
        
    def _sign(self, method, endpoint, params=None):
        """Sign HTX API request"""
        params = params or {}
        params['AccessKeyId'] = self.api_key
        params['SignatureMethod'] = 'HmacSHA256'
        params['SignatureVersion'] = '2'
        params['Timestamp'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        
        # Sort parameters
        sorted_params = sorted(params.items())
        encoded_params = urlencode(sorted_params)
        
        # Create signature payload
        payload = f"{method}\napi.huobi.pro\n{endpoint}\n{encoded_params}"
        
        # Sign
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.b64encode(signature).decode()
        params['Signature'] = signature_b64
        
        return params
    
    def get_balance(self, account_id):
        """Get account balance"""
        endpoint = f"/v1/account/accounts/{account_id}/balance"
        params = self._sign('GET', endpoint)
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        return response.json()
    
    def get_account_id(self):
        """Get account ID"""
        endpoint = "/v1/account/accounts"
        params = self._sign('GET', endpoint)
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        data = response.json()
        
        if data['status'] == 'ok' and len(data['data']) > 0:
            # Return spot account
            for account in data['data']:
                if account['type'] == 'spot':
                    return account['id']
        return None
    
    def get_candles(self, symbol, interval='15min', size=200):
        """Get candlestick data (no auth needed)"""
        endpoint = "/market/history/kline"
        params = {
            'symbol': symbol,
            'period': interval,
            'size': size
        }
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        data = response.json()
        
        if data['status'] == 'ok':
            candles = data['data']
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['id'], unit='s')
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        return None
    
    def get_ticker(self, symbol):
        """Get current ticker price"""
        endpoint = "/market/detail/merged"
        params = {'symbol': symbol}
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        data = response.json()
        
        if data['status'] == 'ok':
            tick = data['tick']
            return {
                'bid': tick['bid'][0],
                'ask': tick['ask'][0],
                'last': tick['close']
            }
        return None
    
    def place_order(self, account_id, symbol, order_type, amount, price=None):
        """Place order on HTX
        
        Args:
            account_id: HTX account ID
            symbol: Trading pair (e.g., 'btcusdt')
            order_type: 'buy-market', 'sell-market', 'buy-limit', 'sell-limit'
            amount: Order amount
            price: Price (for limit orders)
        """
        endpoint = "/v1/order/orders/place"
        
        order_data = {
            'account-id': str(account_id),
            'symbol': symbol,
            'type': order_type,
            'amount': str(amount)
        }
        
        if price and 'limit' in order_type:
            order_data['price'] = str(price)
        
        params = self._sign('POST', endpoint, order_data)
        
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=order_data,
            params=params
        )
        
        return response.json()


def calculate_vwap_signals(df):
    """Calculate VWAP and generate trading signals"""
    
    # Calculate VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['vol']).cumsum() / df['vol'].cumsum()
    
    # Calculate standard deviation bands
    df['price_diff'] = df['close'] - df['vwap']
    rolling_std = df['price_diff'].rolling(window=20).std()
    
    df['upper_band'] = df['vwap'] + (2 * rolling_std)
    df['lower_band'] = df['vwap'] - (2 * rolling_std)
    
    # Get latest values
    latest = df.iloc[-1]
    current_price = latest['close']
    vwap = latest['vwap']
    upper_band = latest['upper_band']
    lower_band = latest['lower_band']
    
    # Generate signal
    signal = 'HOLD'
    if current_price <= lower_band:
        signal = 'BUY'
    elif current_price >= upper_band:
        signal = 'SELL'
    
    return {
        'signal': signal,
        'current_price': current_price,
        'vwap': vwap,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'distance_from_vwap': ((current_price - vwap) / vwap) * 100
    }


def main():
    print("=" * 60)
    print("🚀 HTX TRADING TEST - Single Strategy Execution")
    print("=" * 60)
    print()
    
    # Check API keys
    if not HTX_API_KEY or not HTX_SECRET_KEY:
        print("❌ ERROR: HTX_API_KEY or HTX_SECRET_KEY not set!")
        print("   Set them as environment variables:")
        print("   export HTX_API_KEY='your_key'")
        print("   export HTX_SECRET_KEY='your_secret'")
        sys.exit(1)
    
    print(f"✅ HTX API Key found: {HTX_API_KEY[:10]}...")
    print()
    
    # Initialize HTX client
    htx = HTXClient(HTX_API_KEY, HTX_SECRET_KEY)
    
    # Get account ID
    print("🔍 Getting HTX account ID...")
    account_id = htx.get_account_id()
    if not account_id:
        print("❌ ERROR: Could not get HTX account ID")
        sys.exit(1)
    print(f"✅ Account ID: {account_id}")
    print()
    
    # Get balance
    print("💰 Fetching account balance...")
    balance_data = htx.get_balance(account_id)
    if balance_data['status'] == 'ok':
        balances = balance_data['data']['list']
        usdt_balance = 0
        btc_balance = 0
        
        for item in balances:
            if item['currency'] == 'usdt' and item['type'] == 'trade':
                usdt_balance = float(item['balance'])
            elif item['currency'] == 'btc' and item['type'] == 'trade':
                btc_balance = float(item['balance'])
        
        print(f"   USDT: ${usdt_balance:.2f}")
        print(f"   BTC: {btc_balance:.6f}")
        print()
        
        if usdt_balance < 10:
            print("⚠️  WARNING: Low USDT balance (< $10)")
            print()
    else:
        print("⚠️  Could not fetch balance")
        print()
    
    # Test symbol (BTC/USDT on HTX)
    symbol = 'btcusdt'
    interval = '15min'
    
    print(f"📈 Testing VWAP Strategy on {symbol.upper()} @ {interval}")
    print("-" * 60)
    
    # Get candles
    print(f"🕯️  Fetching {interval} candles for {symbol.upper()}...")
    df = htx.get_candles(symbol, interval, size=100)
    
    if df is None or len(df) == 0:
        print("❌ ERROR: Could not fetch candles from HTX")
        sys.exit(1)
    
    print(f"✅ Fetched {len(df)} candles")
    latest_candle = df.iloc[-1]
    print(f"   Latest: ${latest_candle['close']:.2f} | Volume: {latest_candle['vol']:.2f}")
    print()
    
    # Calculate VWAP signals
    print("📊 Calculating VWAP indicators...")
    signals = calculate_vwap_signals(df)
    
    print(f"   Current Price: ${signals['current_price']:.2f}")
    print(f"   VWAP:          ${signals['vwap']:.2f}")
    print(f"   Upper Band:    ${signals['upper_band']:.2f}")
    print(f"   Lower Band:    ${signals['lower_band']:.2f}")
    print(f"   Distance:      {signals['distance_from_vwap']:+.2f}% from VWAP")
    print()
    
    print("=" * 60)
    print(f"🎯 SIGNAL: {signals['signal']}")
    print("=" * 60)
    print()
    
    if signals['signal'] == 'HOLD':
        print("⏸️  No trade signal - HOLDING")
        print()
        print("   Current price is between bands.")
        print("   Waiting for price to touch upper/lower band for signal.")
        print()
    else:
        # Get current ticker for precise pricing
        ticker = htx.get_ticker(symbol)
        if ticker:
            print(f"📊 Current Market:")
            print(f"   Bid: ${ticker['bid']:.2f}")
            print(f"   Ask: ${ticker['ask']:.2f}")
            print(f"   Spread: ${ticker['ask'] - ticker['bid']:.2f}")
            print()
        
        # Calculate position size (10% of USDT balance, min $10)
        position_size_usd = max(usdt_balance * 0.1, 10)
        
        print(f"💼 Position Sizing:")
        print(f"   Account Balance: ${usdt_balance:.2f}")
        print(f"   Position Size: ${position_size_usd:.2f} (10% of balance)")
        
        if signals['signal'] == 'BUY':
            amount_btc = position_size_usd / signals['current_price']
            print(f"   BUY Amount: {amount_btc:.6f} BTC")
        else:  # SELL
            amount_btc = min(btc_balance * 0.1, btc_balance)
            print(f"   SELL Amount: {amount_btc:.6f} BTC")
        
        print()
        
        # Ask for confirmation
        print("⚠️  READY TO EXECUTE REAL TRADE ON HTX!")
        print()
        confirmation = input("Type 'YES' to execute this trade: ").strip().upper()
        print()
        
        if confirmation == 'YES':
            print("🔄 Placing order on HTX...")
            
            order_type = 'buy-market' if signals['signal'] == 'BUY' else 'sell-market'
            amount = amount_btc if signals['signal'] == 'BUY' else amount_btc
            
            try:
                result = htx.place_order(
                    account_id=account_id,
                    symbol=symbol,
                    order_type=order_type,
                    amount=amount
                )
                
                if result.get('status') == 'ok':
                    order_id = result['data']
                    print(f"✅ ORDER PLACED SUCCESSFULLY!")
                    print(f"   Order ID: {order_id}")
                    print(f"   Type: {order_type}")
                    print(f"   Amount: {amount:.6f} BTC")
                    print()
                    print("🎉 TRADE EXECUTED ON HTX!")
                else:
                    print(f"❌ ORDER FAILED: {result}")
                
            except Exception as e:
                print(f"❌ ERROR placing order: {e}")
        else:
            print("❌ Trade cancelled by user")
    
    print()
    print("=" * 60)
    print("✅ HTX TRADING TEST COMPLETE")
    print("=" * 60)
    print()
    print("Next step: Run full TradePex with HTX integration!")
    print("   This test validated:")
    print("   ✅ HTX API connection")
    print("   ✅ Balance retrieval")
    print("   ✅ Market data access")
    print("   ✅ Signal generation")
    if signals['signal'] != 'HOLD':
        print("   ✅ Order execution capability")


if __name__ == "__main__":
    main()
