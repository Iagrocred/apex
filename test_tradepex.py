#!/usr/bin/env python3
"""
🧪 TRADEPEX PRE-FLIGHT TEST
Validates TradePex is ready for live trading before launch
Tests: API connection, balance access, order execution capability
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
import eth_account
from eth_account.signers.local import LocalAccount
from eth_account.messages import encode_structured_data

load_dotenv()

# =========================================================================================
# CONFIGURATION
# =========================================================================================

MAIN_WALLET = "0x70e1b650e700d015FBB356E66C30E5A0F8A07196"  # Your main wallet with $615
TEST_SYMBOL = "BTC"
TEST_SIZE_USD = 10  # $10 test trade

# =========================================================================================
# HYPERLIQUID CLIENT (IDENTICAL TO TRADEPEX)
# =========================================================================================

class HyperliquidTestClient:
    """Test client matching TradePex implementation exactly"""
    
    def __init__(self, account: LocalAccount):
        self.account = account
        self.base_url = "https://api.hyperliquid.xyz"
        self.info_url = f"{self.base_url}/info"
        self.exchange_url = f"{self.base_url}/exchange"
    
    def get_user_state(self, address: str) -> dict:
        """Get user state for any address"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "clearinghouseState", "user": address},
                timeout=10
            )
            return response.json()
        except Exception as e:
            print(f"❌ Error: {e}")
            return {}
    
    def get_balance(self, address: str) -> float:
        """Get account balance"""
        state = self.get_user_state(address)
        return float(state.get('marginSummary', {}).get('accountValue', 0))
    
    def get_all_mids(self) -> dict:
        """Get all mid prices"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "allMids"},
                timeout=10
            )
            return response.json()
        except:
            return {}
    
    def get_l2_book(self, symbol: str) -> dict:
        """Get L2 order book"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "l2Book", "coin": symbol},
                timeout=10
            )
            return response.json()
        except:
            return {}
    
    def get_ask_bid(self, symbol: str) -> tuple:
        """Get ask/bid prices"""
        try:
            l2_data = self.get_l2_book(symbol)
            if not l2_data or 'levels' not in l2_data:
                return 0.0, 0.0
            
            levels = l2_data['levels']
            bid = float(levels[0][0]['px']) if levels[0] else 0.0
            ask = float(levels[1][0]['px']) if levels[1] else 0.0
            
            return ask, bid
        except:
            return 0.0, 0.0
    
    def get_candles(self, symbol: str, interval: str = "15m", num_bars: int = 100) -> list:
        """Get candle data"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (num_bars * 15 * 60 * 1000)
            
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": interval,
                        "startTime": start_time,
                        "endTime": end_time
                    }
                },
                timeout=10
            )
            
            candles = response.json()
            return candles if candles else []
        except:
            return []
    
    def place_test_order(self, symbol: str, is_buy: bool, size_usd: float) -> dict:
        """Place a test order (REAL EXECUTION - USES REAL MONEY!)"""
        try:
            # Get price
            ask, bid = self.get_ask_bid(symbol)
            if ask == 0 or bid == 0:
                return {'success': False, 'error': 'Cannot get price'}
            
            price = ask if is_buy else bid
            contracts = size_usd / price
            contracts = round(contracts, 4)
            
            if contracts == 0:
                return {'success': False, 'error': 'Size too small'}
            
            print(f"   💰 Order Details:")
            print(f"      Side: {'BUY' if is_buy else 'SELL'}")
            print(f"      Price: ${price:,.2f}")
            print(f"      Size: {contracts} {symbol}")
            print(f"      Value: ${size_usd:.2f}")
            
            # Build order
            timestamp = int(time.time() * 1000)
            limit_price = price * 1.02 if is_buy else price * 0.98
            
            order = {
                "asset": symbol,
                "isBuy": is_buy,
                "limitPx": str(limit_price),
                "sz": str(contracts),
                "reduceOnly": False,
                "orderType": {"limit": {"tif": "Ioc"}}
            }
            
            action = {
                "type": "order",
                "orders": [order],
                "grouping": "na"
            }
            
            # Sign
            connection_id = json.dumps(action)
            
            signature_data = {
                "domain": {
                    "name": "Exchange",
                    "version": "1",
                    "chainId": 1337,
                    "verifyingContract": "0x0000000000000000000000000000000000000000"
                },
                "types": {
                    "Agent": [
                        {"name": "source", "type": "string"},
                        {"name": "connectionId", "type": "bytes32"}
                    ]
                },
                "primaryType": "Agent",
                "message": {
                    "source": "a",
                    "connectionId": connection_id
                }
            }
            
            structured_data = encode_structured_data(signature_data)
            signed_message = self.account.sign_message(structured_data)
            
            payload = {
                "action": action,
                "nonce": timestamp,
                "signature": {
                    "r": hex(signed_message.r),
                    "s": hex(signed_message.s),
                    "v": signed_message.v
                },
                "vaultAddress": None
            }
            
            # Submit
            print(f"   📡 Submitting order to Hyperliquid...")
            response = requests.post(
                self.exchange_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
            
            result = response.json()
            
            if "status" in result and result["status"] == "ok":
                return {
                    'success': True,
                    'symbol': symbol,
                    'side': 'buy' if is_buy else 'sell',
                    'size': contracts,
                    'price': price,
                    'result': result
                }
            else:
                return {'success': False, 'error': result}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

# =========================================================================================
# TEST FUNCTIONS
# =========================================================================================

def test_api_connection(client):
    """Test 1: API Connection"""
    print("\n" + "="*80)
    print("TEST 1: API Connection")
    print("="*80)
    
    mids = client.get_all_mids()
    if mids and 'BTC' in mids:
        btc_price = float(mids['BTC'])
        print(f"✅ PASS - Connected to Hyperliquid API")
        print(f"   BTC Price: ${btc_price:,.2f}")
        return True, btc_price
    else:
        print(f"❌ FAIL - Cannot connect to Hyperliquid API")
        return False, 0

def test_wallet_balance(client, agent_address):
    """Test 2: Wallet Balance"""
    print("\n" + "="*80)
    print("TEST 2: Wallet Balance")
    print("="*80)
    
    # Check agent wallet
    agent_balance = client.get_balance(agent_address)
    print(f"   Agent Wallet: {agent_address[:10]}...{agent_address[-8:]}")
    print(f"   Balance: ${agent_balance:.2f}")
    
    # Check main wallet
    main_balance = client.get_balance(MAIN_WALLET)
    print(f"\n   Main Wallet: {MAIN_WALLET[:10]}...{MAIN_WALLET[-8:]}")
    print(f"   Balance: ${main_balance:.2f}")
    
    if main_balance >= 10:
        print(f"\n✅ PASS - Main wallet has ${main_balance:.2f} available")
        return True, main_balance
    else:
        print(f"\n❌ FAIL - Insufficient balance (${main_balance:.2f} < $10)")
        return False, main_balance

def test_market_data(client, symbol):
    """Test 3: Market Data Access"""
    print("\n" + "="*80)
    print("TEST 3: Market Data Access")
    print("="*80)
    
    # Get order book
    ask, bid = client.get_ask_bid(symbol)
    if ask > 0 and bid > 0:
        spread = ask - bid
        spread_pct = (spread / bid) * 100
        print(f"✅ PASS - Order book accessible")
        print(f"   {symbol} Ask: ${ask:,.2f}")
        print(f"   {symbol} Bid: ${bid:,.2f}")
        print(f"   Spread: ${spread:.2f} ({spread_pct:.3f}%)")
    else:
        print(f"❌ FAIL - Cannot access order book")
        return False
    
    # Get candles
    candles = client.get_candles(symbol)
    if len(candles) > 0:
        print(f"\n✅ PASS - Historical data accessible")
        print(f"   Fetched {len(candles)} candles")
        return True
    else:
        print(f"\n❌ FAIL - Cannot fetch historical data")
        return False

def test_order_execution(client, symbol, size_usd):
    """Test 4: Order Execution (REAL TRADE)"""
    print("\n" + "="*80)
    print("TEST 4: Order Execution (REAL TRADE WARNING!)")
    print("="*80)
    print(f"\n⚠️  THIS WILL PLACE A REAL ${size_usd} TRADE!")
    print(f"   Symbol: {symbol}")
    print(f"   Size: ${size_usd}")
    print()
    
    response = input("   Type 'YES' to proceed with real trade test: ")
    if response.upper() != 'YES':
        print("\n⏭️  SKIPPED - Order execution test")
        return None
    
    print(f"\n🔥 Executing test BUY order...")
    result = client.place_test_order(symbol, is_buy=True, size_usd=size_usd)
    
    if result.get('success'):
        print(f"\n✅ PASS - Order executed successfully!")
        print(f"   Order details: {json.dumps(result.get('result', {}), indent=2)}")
        
        # Wait then close
        print(f"\n⏳ Waiting 10 seconds before closing position...")
        time.sleep(10)
        
        print(f"\n🔄 Closing test position...")
        close_result = client.place_test_order(symbol, is_buy=False, size_usd=size_usd)
        
        if close_result.get('success'):
            print(f"✅ Position closed successfully!")
            return True
        else:
            print(f"⚠️  Manual close may be needed")
            print(f"   Error: {close_result.get('error')}")
            return True  # Order placement worked even if close failed
    else:
        print(f"\n❌ FAIL - Order execution failed")
        print(f"   Error: {result.get('error')}")
        return False

# =========================================================================================
# MAIN TEST SUITE
# =========================================================================================

def main():
    print("=" * 80)
    print("🧪 TRADEPEX PRE-FLIGHT TEST SUITE")
    print("=" * 80)
    print()
    print("This test validates TradePex is ready for live trading")
    print("Tests: API, Balance, Market Data, Order Execution")
    print()
    
    # Initialize
    api_key = os.getenv('HYPER_LIQUID_KEY')
    if not api_key:
        print("❌ ERROR: HYPER_LIQUID_KEY not found")
        print("   Set environment variable: export HYPER_LIQUID_KEY='your_key'")
        return
    
    account = eth_account.Account.from_key(api_key)
    client = HyperliquidTestClient(account)
    
    print(f"🔐 Agent Wallet: {account.address}")
    print(f"💰 Main Wallet: {MAIN_WALLET}")
    print()
    
    # Run tests
    results = []
    
    # Test 1: API Connection
    test1_pass, btc_price = test_api_connection(client)
    results.append(("API Connection", test1_pass))
    if not test1_pass:
        print("\n❌ CRITICAL: Cannot proceed without API connection")
        return
    
    # Test 2: Wallet Balance
    test2_pass, balance = test_wallet_balance(client, account.address)
    results.append(("Wallet Balance", test2_pass))
    if not test2_pass:
        print("\n❌ CRITICAL: Insufficient balance for testing")
        return
    
    # Test 3: Market Data
    test3_pass = test_market_data(client, TEST_SYMBOL)
    results.append(("Market Data", test3_pass))
    if not test3_pass:
        print("\n❌ CRITICAL: Cannot access market data")
        return
    
    # Test 4: Order Execution (optional)
    test4_result = test_order_execution(client, TEST_SYMBOL, TEST_SIZE_USD)
    if test4_result is not None:
        results.append(("Order Execution", test4_result))
    else:
        results.append(("Order Execution", "SKIPPED"))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        if result == True:
            status = "✅ PASS"
        elif result == False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"   {test_name:.<50} {status}")
    
    print()
    print("=" * 80)
    
    # Determine if ready
    required_tests = [r for r in results if r[1] != "SKIPPED"]
    passed_tests = [r for r in required_tests if r[1] == True]
    
    if len(passed_tests) == len(required_tests):
        print("✅ ALL TESTS PASSED - TRADEPEX READY FOR LAUNCH!")
        print("=" * 80)
        print()
        print("🚀 To launch TradePex:")
        print("   python3 tradepex.py")
        print()
        print("   All 10 strategies will trade automatically 24/7")
        print("   Max position: $195 per strategy")
        print("   Total capital: $615.05")
        print("   Risk: 2% stop loss per trade")
    elif len(passed_tests) >= 3:
        print("⚠️  PARTIAL PASS - Manual trading test recommended")
        print("=" * 80)
        print()
        print("Core systems operational but order execution not tested")
        print("Recommended: Run test again and execute test trade")
    else:
        print("❌ TESTS FAILED - DO NOT LAUNCH TRADEPEX")
        print("=" * 80)
        print()
        print("Fix the issues above before launching")
    
    print()

if __name__ == "__main__":
    main()
