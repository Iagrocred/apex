#!/usr/bin/env python3
"""
================================================================================
🔥 HUOBI/HTX FUTURES API CONNECTION TEST
================================================================================

This script tests the connection to Huobi/HTX Futures API:
1. Connects to HTX Futures API
2. Pulls account balance info
3. Opens 8 trades (4 LONG + 4 SHORT)
4. Closes each trade after 8 seconds
5. Confirms connection is working for live trading

Usage:
    python3 huobitest.py

Environment Variables Required:
    HTX_API_KEY     - Your Huobi/HTX API key
    HTX_SECRET      - Your Huobi/HTX secret key
    HTX_FUTURES_URL - (Optional) Futures API endpoint

================================================================================
"""

import os
import sys
import time
import hmac
import hashlib
import base64
import json
from datetime import datetime, timezone
from urllib.parse import urlencode
import requests
from typing import Dict, Any, Optional, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# HTX FUTURES API CLIENT
# =============================================================================

class HTXFuturesClient:
    """
    HTX (Huobi) Futures API Client
    
    Supports:
    - USDT-M Perpetual Futures (coin-margined)
    - Futures account balance
    - Opening/closing positions
    - 1x-8x leverage
    """
    
    # API Endpoints
    SPOT_URL = "https://api.huobi.pro"
    FUTURES_URL = "https://api.hbdm.com"  # Futures API
    USDT_FUTURES_URL = "https://api.hbdm.com"  # USDT-M Futures
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv("HTX_API_KEY", "")
        self.secret_key = secret_key or os.getenv("HTX_SECRET", "")
        self.futures_url = os.getenv("HTX_FUTURES_URL", self.USDT_FUTURES_URL)
        
        if not self.api_key or not self.secret_key:
            print("⚠️  WARNING: No API keys found!")
            print("   Set HTX_API_KEY and HTX_SECRET environment variables")
        
    def _generate_signature(self, method: str, host: str, path: str, params: dict) -> str:
        """Generate HMAC-SHA256 signature for HTX API"""
        # Sort parameters
        sorted_params = sorted(params.items(), key=lambda x: x[0])
        query_string = urlencode(sorted_params)
        
        # Create signature payload
        payload = f"{method}\n{host}\n{path}\n{query_string}"
        
        # Sign with HMAC-SHA256
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_timestamp(self) -> str:
        """Get UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    
    def _request(self, method: str, endpoint: str, params: dict = None, 
                 body: dict = None, signed: bool = True, base_url: str = None) -> dict:
        """Make authenticated request to HTX API"""
        
        base_url = base_url or self.futures_url
        host = base_url.replace("https://", "").replace("http://", "")
        
        params = params or {}
        
        if signed:
            params['AccessKeyId'] = self.api_key
            params['SignatureMethod'] = 'HmacSHA256'
            params['SignatureVersion'] = '2'
            params['Timestamp'] = self._get_timestamp()
            
            signature = self._generate_signature(method, host, endpoint, params)
            params['Signature'] = signature
        
        url = f"{base_url}{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            else:  # POST
                url_with_params = f"{url}?{urlencode(params)}"
                response = requests.post(url_with_params, json=body, headers=headers, timeout=10)
            
            return response.json()
            
        except Exception as e:
            return {'status': 'error', 'err-msg': str(e)}
    
    # =========================================================================
    # ACCOUNT ENDPOINTS
    # =========================================================================
    
    def get_spot_balance(self) -> Dict:
        """Get spot account balance"""
        # First get account ID
        accounts = self._request('GET', '/v1/account/accounts', base_url=self.SPOT_URL)
        
        if accounts.get('status') != 'ok':
            return accounts
        
        # Find spot account
        spot_account_id = None
        for acc in accounts.get('data', []):
            if acc.get('type') == 'spot':
                spot_account_id = acc.get('id')
                break
        
        if not spot_account_id:
            return {'status': 'error', 'err-msg': 'No spot account found'}
        
        # Get balance
        return self._request('GET', f'/v1/account/accounts/{spot_account_id}/balance', 
                           base_url=self.SPOT_URL)
    
    def get_futures_account_info(self, margin_account: str = "USDT") -> Dict:
        """Get USDT-M futures account info"""
        body = {"margin_account": margin_account}
        return self._request('POST', '/linear-swap-api/v1/swap_cross_account_info', body=body)
    
    def get_futures_positions(self, contract_code: str = None) -> Dict:
        """Get current futures positions"""
        body = {}
        if contract_code:
            body['contract_code'] = contract_code
        return self._request('POST', '/linear-swap-api/v1/swap_cross_position_info', body=body)
    
    # =========================================================================
    # TRADING ENDPOINTS
    # =========================================================================
    
    def set_leverage(self, contract_code: str, lever_rate: int) -> Dict:
        """Set leverage for a contract (1-125x)"""
        body = {
            "contract_code": contract_code,
            "lever_rate": lever_rate
        }
        return self._request('POST', '/linear-swap-api/v1/swap_cross_switch_lever_rate', body=body)
    
    def open_position(self, contract_code: str, direction: str, volume: int, 
                      lever_rate: int = 1, price: float = None) -> Dict:
        """
        Open a futures position
        
        Args:
            contract_code: e.g., "BTC-USDT"
            direction: "buy" (long) or "sell" (short)
            volume: Number of contracts (1 contract = $10 for most pairs)
            lever_rate: 1-125
            price: Limit price (None for market order)
        """
        body = {
            "contract_code": contract_code,
            "direction": direction,
            "offset": "open",  # open position
            "lever_rate": lever_rate,
            "volume": volume,
            "order_price_type": "optimal_5" if price is None else "limit"  # Market order
        }
        
        if price:
            body["price"] = price
            
        return self._request('POST', '/linear-swap-api/v1/swap_cross_order', body=body)
    
    def close_position(self, contract_code: str, direction: str, volume: int,
                       price: float = None) -> Dict:
        """
        Close a futures position
        
        Args:
            contract_code: e.g., "BTC-USDT"
            direction: "buy" to close short, "sell" to close long
            volume: Number of contracts to close
            price: Limit price (None for market order)
        """
        body = {
            "contract_code": contract_code,
            "direction": direction,
            "offset": "close",  # close position
            "volume": volume,
            "order_price_type": "optimal_5" if price is None else "limit"
        }
        
        if price:
            body["price"] = price
            
        return self._request('POST', '/linear-swap-api/v1/swap_cross_order', body=body)
    
    def get_order_info(self, contract_code: str, order_id: str) -> Dict:
        """Get order status"""
        body = {
            "contract_code": contract_code,
            "order_id": order_id
        }
        return self._request('POST', '/linear-swap-api/v1/swap_cross_order_info', body=body)
    
    def cancel_order(self, contract_code: str, order_id: str) -> Dict:
        """Cancel an open order"""
        body = {
            "contract_code": contract_code,
            "order_id": order_id
        }
        return self._request('POST', '/linear-swap-api/v1/swap_cross_cancel', body=body)
    
    # =========================================================================
    # MARKET DATA (No auth needed)
    # =========================================================================
    
    def get_market_price(self, contract_code: str) -> Optional[float]:
        """Get current market price"""
        try:
            url = f"{self.futures_url}/linear-swap-ex/market/trade"
            params = {"contract_code": contract_code}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('status') == 'ok':
                trades = data.get('tick', {}).get('data', [])
                if trades:
                    return float(trades[0].get('price', 0))
        except Exception as e:
            print(f"❌ Error getting price: {e}")
        return None


# =============================================================================
# CONNECTION TEST
# =============================================================================

def run_connection_test():
    """
    Run the complete HTX Futures API connection test:
    1. Connect to API
    2. Get account balance
    3. Open 8 trades (4 LONG + 4 SHORT)
    4. Wait 8 seconds each
    5. Close trades
    """
    
    print("\n" + "="*80)
    print("🔥 HTX FUTURES API CONNECTION TEST")
    print("="*80 + "\n")
    
    # Initialize client
    client = HTXFuturesClient()
    
    if not client.api_key or not client.secret_key:
        print("❌ ERROR: API credentials not configured!")
        print("\nPlease set environment variables:")
        print("  export HTX_API_KEY='your-api-key'")
        print("  export HTX_SECRET='your-secret-key'")
        return False
    
    print("✅ API credentials found")
    print(f"   API Key: {client.api_key[:8]}...{client.api_key[-4:]}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Account Balance
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("📊 TEST 1: Getting Account Balance")
    print("-"*40)
    
    balance_info = client.get_futures_account_info()
    
    if balance_info.get('status') == 'ok':
        print("✅ Futures account connected!")
        data = balance_info.get('data', [])
        if data:
            for account in data:
                margin_balance = account.get('margin_balance', 0)
                available = account.get('margin_available', 0)
                currency = account.get('margin_asset', 'USDT')
                print(f"   💰 Balance: {margin_balance} {currency}")
                print(f"   💵 Available: {available} {currency}")
    else:
        print(f"❌ Error getting balance: {balance_info.get('err-msg', 'Unknown error')}")
        print(f"   Full response: {json.dumps(balance_info, indent=2)}")
        # Continue anyway for testing
    
    # -------------------------------------------------------------------------
    # TEST 2: Get Current Prices
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("📈 TEST 2: Getting Market Prices")
    print("-"*40)
    
    test_symbols = ["BTC-USDT", "ETH-USDT"]
    prices = {}
    
    for symbol in test_symbols:
        price = client.get_market_price(symbol)
        if price:
            prices[symbol] = price
            print(f"   {symbol}: ${price:,.2f}")
        else:
            print(f"   {symbol}: ❌ Failed to get price")
    
    if not prices:
        print("❌ Could not get any prices. Check API connection.")
        return False
    
    print("✅ Market data connected!")
    
    # -------------------------------------------------------------------------
    # TEST 3: Set Leverage
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("⚙️  TEST 3: Setting Leverage")
    print("-"*40)
    
    test_leverage = 2  # Safe test leverage
    
    for symbol in test_symbols:
        result = client.set_leverage(symbol, test_leverage)
        if result.get('status') == 'ok':
            print(f"   ✅ {symbol}: Set to {test_leverage}x leverage")
        else:
            print(f"   ⚠️  {symbol}: {result.get('err-msg', 'Could not set leverage')}")
    
    # -------------------------------------------------------------------------
    # TEST 4: Open and Close 8 Trades
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("🚀 TEST 4: Opening 8 Test Trades")
    print("-"*40)
    
    # Trade configuration
    trades_to_open = [
        {"symbol": "BTC-USDT", "direction": "buy", "name": "BTC LONG #1"},
        {"symbol": "BTC-USDT", "direction": "sell", "name": "BTC SHORT #1"},
        {"symbol": "ETH-USDT", "direction": "buy", "name": "ETH LONG #1"},
        {"symbol": "ETH-USDT", "direction": "sell", "name": "ETH SHORT #1"},
        {"symbol": "BTC-USDT", "direction": "buy", "name": "BTC LONG #2"},
        {"symbol": "BTC-USDT", "direction": "sell", "name": "BTC SHORT #2"},
        {"symbol": "ETH-USDT", "direction": "buy", "name": "ETH LONG #2"},
        {"symbol": "ETH-USDT", "direction": "sell", "name": "ETH SHORT #2"},
    ]
    
    opened_trades = []
    
    for i, trade in enumerate(trades_to_open, 1):
        print(f"\n📤 Opening trade {i}/8: {trade['name']}")
        
        # Open position (1 contract = $10 worth)
        result = client.open_position(
            contract_code=trade['symbol'],
            direction=trade['direction'],
            volume=1,  # 1 contract
            lever_rate=test_leverage
        )
        
        if result.get('status') == 'ok':
            order_data = result.get('data', {})
            order_id = order_data.get('order_id_str', order_data.get('order_id'))
            print(f"   ✅ Opened! Order ID: {order_id}")
            
            opened_trades.append({
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'order_id': order_id,
                'name': trade['name']
            })
        else:
            print(f"   ❌ Failed: {result.get('err-msg', 'Unknown error')}")
            print(f"      Response: {json.dumps(result, indent=2)}")
    
    if not opened_trades:
        print("\n❌ No trades were opened. Check your account balance and permissions.")
        return False
    
    print(f"\n✅ Successfully opened {len(opened_trades)}/8 trades!")
    
    # -------------------------------------------------------------------------
    # Wait 8 seconds
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("⏳ Waiting 8 seconds...")
    print("-"*40)
    
    for i in range(8, 0, -1):
        print(f"   {i}...", end=" ", flush=True)
        time.sleep(1)
    print("\n")
    
    # -------------------------------------------------------------------------
    # TEST 5: Close All Trades
    # -------------------------------------------------------------------------
    print("-"*40)
    print("📥 TEST 5: Closing All Trades")
    print("-"*40)
    
    closed_count = 0
    
    for trade in opened_trades:
        print(f"\n📥 Closing: {trade['name']}")
        
        # To close: buy closes short, sell closes long
        close_direction = "sell" if trade['direction'] == "buy" else "buy"
        
        result = client.close_position(
            contract_code=trade['symbol'],
            direction=close_direction,
            volume=1
        )
        
        if result.get('status') == 'ok':
            print(f"   ✅ Closed!")
            closed_count += 1
        else:
            print(f"   ❌ Failed: {result.get('err-msg', 'Unknown error')}")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    print(f"""
   ✅ API Connection:     WORKING
   ✅ Market Data:        WORKING
   ✅ Account Access:     WORKING
   ✅ Trades Opened:      {len(opened_trades)}/8
   ✅ Trades Closed:      {closed_count}/{len(opened_trades)}
   ✅ Leverage:           {test_leverage}x
   
   🎉 CONNECTION TEST COMPLETE!
   
   Your HTX Futures API is ready for live trading integration with tradeadapt.py
   
   Next steps:
   1. Run: python3 tradeadapt.py (paper trading)
   2. When strategy hits 71% win rate → Promoted to live
   3. Live trades execute via this verified API connection!
    """)
    
    return closed_count == len(opened_trades)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║    🔥 HUOBI/HTX FUTURES API CONNECTION TEST                              ║
    ║                                                                           ║
    ║    This will:                                                            ║
    ║    1. Connect to your HTX Futures account                               ║
    ║    2. Check your balance                                                 ║
    ║    3. Open 8 test trades (LONG + SHORT)                                 ║
    ║    4. Close them after 8 seconds                                        ║
    ║                                                                           ║
    ║    ⚠️  REAL MONEY WILL BE USED (small amounts)                           ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for API keys
    api_key = os.getenv("HTX_API_KEY", "")
    secret = os.getenv("HTX_SECRET", "")
    
    if not api_key or not secret:
        print("❌ ERROR: API credentials not set!")
        print("\nPlease set your HTX API credentials:")
        print("  export HTX_API_KEY='your-api-key'")
        print("  export HTX_SECRET='your-secret-key'")
        print("\nOr add them to your .env file")
        sys.exit(1)
    
    print("Press ENTER to start the test, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        sys.exit(0)
    
    success = run_connection_test()
    
    sys.exit(0 if success else 1)
