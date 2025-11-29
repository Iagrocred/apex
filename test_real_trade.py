#!/usr/bin/env python3
"""
🚀 REAL TRADING EXECUTION TEST
Tests actual order placement and execution on Hyperliquid
Uses MINIMAL position ($10) to verify trading works
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

# Your wallet addresses
MAIN_WALLET = "0x70e1b650e700d015FBB356E66C30E5A0F8A07196"  # Has $615.05
# Agent wallet will be derived from API key (used for signing only)

# Test trade parameters
TEST_SYMBOL = "BTC"  # Symbol to trade
TEST_SIZE = 0.0001   # Tiny size (~$8-10 worth of BTC)
TEST_LEVERAGE = 2    # 2x leverage for test

# =========================================================================================
# HYPERLIQUID CLIENT
# =========================================================================================

class HyperliquidClient:
    """Full Hyperliquid client with order execution"""
    
    def __init__(self, account: LocalAccount):
        self.account = account
        self.base_url = "https://api.hyperliquid.xyz"
        self.info_url = f"{self.base_url}/info"
        self.exchange_url = f"{self.base_url}/exchange"
        
        print(f"✅ Hyperliquid Client Initialized")
        print(f"   Agent Wallet: {account.address[:6]}...{account.address[-4:]}")
        print(f"   (Used for signing transactions)")
    
    def get_user_state(self, address: str) -> dict:
        """Get complete user state for any address"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "clearinghouseState", "user": address},
                timeout=10
            )
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching user state: {e}")
            return {}
    
    def get_balance(self, address: str) -> float:
        """Get account balance for any address"""
        try:
            state = self.get_user_state(address)
            balance = float(state.get('marginSummary', {}).get('accountValue', 0))
            return balance
        except:
            return 0.0
    
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
            return float(all_mids.get(symbol, 0))
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return 0.0
    
    def get_meta(self) -> dict:
        """Get exchange metadata"""
        try:
            response = requests.post(
                self.info_url,
                headers={"Content-Type": "application/json"},
                json={"type": "meta"},
                timeout=10
            )
            return response.json()
        except:
            return {}
    
    def place_market_order(self, symbol: str, is_buy: bool, size: float, reduce_only: bool = False) -> dict:
        """Place a market order"""
        try:
            print(f"\n📝 Preparing {'BUY' if is_buy else 'SELL'} order:")
            print(f"   Symbol: {symbol}")
            print(f"   Size: {size}")
            print(f"   Type: Market")
            
            # Get current price for display
            price = self.get_mid_price(symbol)
            print(f"   Current Price: ${price:,.2f}")
            print(f"   Notional: ${price * size:.2f}")
            
            # Build order
            timestamp = int(time.time() * 1000)
            
            order_spec = {
                "order": {
                    "asset": symbol,
                    "isBuy": is_buy,
                    "limitPx": str(price * 1.05 if is_buy else price * 0.95),  # Slippage protection
                    "sz": str(size),
                    "reduceOnly": reduce_only,
                    "orderType": {"limit": {"tif": "Ioc"}}  # Immediate or cancel
                }
            }
            
            # Create action
            action = {
                "type": "order",
                "orders": [order_spec["order"]],
                "grouping": "na"
            }
            
            # Sign action
            connection_id = json.dumps(action)
            
            phantom_agent = {
                "source": "a",
                "connectionId": connection_id
            }
            
            # Create signature
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
                "message": phantom_agent
            }
            
            structured_data = encode_structured_data(signature_data)
            signed_message = self.account.sign_message(structured_data)
            
            # Build request payload
            payload = {
                "action": action,
                "nonce": timestamp,
                "signature": {
                    "r": hex(signed_message.r),
                    "s": hex(signed_message.s),
                    "v": signed_message.v
                }
            }
            
            print(f"\n🚀 Sending order to Hyperliquid...")
            
            # Send order
            response = requests.post(
                self.exchange_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
            
            result = response.json()
            
            print(f"\n📥 Response: {json.dumps(result, indent=2)}")
            
            if "status" in result and result["status"] == "ok":
                print(f"✅ Order placed successfully!")
                return {"success": True, "result": result}
            else:
                print(f"⚠️  Order response: {result}")
                return {"success": False, "result": result}
                
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

# =========================================================================================
# MAIN TEST
# =========================================================================================

def main():
    print("=" * 80)
    print("🚀 REAL TRADING EXECUTION TEST")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will place a REAL TRADE on Hyperliquid!")
    print(f"   Test size: {TEST_SIZE} {TEST_SYMBOL} (~$10)")
    print()
    
    # Initialize
    hyperliquid_key = os.getenv('HYPER_LIQUID_KEY')
    if not hyperliquid_key:
        print("❌ ERROR: HYPER_LIQUID_KEY not found in environment")
        return
    
    account = eth_account.Account.from_key(hyperliquid_key)
    client = HyperliquidClient(account)
    print()
    
    # Check MAIN wallet balance (where the money is)
    print(f"💰 Checking MAIN wallet balance:")
    print(f"   Address: {MAIN_WALLET}")
    main_balance = client.get_balance(MAIN_WALLET)
    print(f"   Balance: ${main_balance:.2f}")
    
    if main_balance < 10:
        print(f"❌ Insufficient balance for test trade")
        return
    
    print(f"✅ Main wallet has funds!")
    print()
    
    # Check agent wallet balance (signing wallet)
    print(f"🔐 Checking AGENT wallet:")
    print(f"   Address: {account.address}")
    agent_balance = client.get_balance(account.address)
    print(f"   Balance: ${agent_balance:.2f}")
    print(f"   (Agent wallet is for signing only)")
    print()
    
    # Get current BTC price
    print(f"📊 Getting {TEST_SYMBOL} market info:")
    btc_price = client.get_mid_price(TEST_SYMBOL)
    print(f"   Current Price: ${btc_price:,.2f}")
    print(f"   Test Trade Value: ${btc_price * TEST_SIZE:.2f}")
    print()
    
    if btc_price == 0:
        print(f"❌ Could not get {TEST_SYMBOL} price")
        return
    
    # Confirm execution
    print("=" * 80)
    print("🎯 READY TO EXECUTE TEST TRADE")
    print("=" * 80)
    print(f"   Action: BUY")
    print(f"   Symbol: {TEST_SYMBOL}")
    print(f"   Size: {TEST_SIZE}")
    print(f"   Value: ${btc_price * TEST_SIZE:.2f}")
    print(f"   Account: Main wallet (${main_balance:.2f})")
    print()
    
    # Execute BUY order
    print("🔥 EXECUTING BUY ORDER...")
    print()
    
    result = client.place_market_order(TEST_SYMBOL, is_buy=True, size=TEST_SIZE)
    
    if result.get("success"):
        print()
        print("=" * 80)
        print("✅ TRADE EXECUTED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("🎉 Trading system is OPERATIONAL!")
        print("   ✅ API connection works")
        print("   ✅ Order signing works")
        print("   ✅ Trade execution works")
        print()
        print("🚀 READY TO RUN FULL TRADEPEX WITH ALL STRATEGIES!")
        print()
        
        # Wait a bit then close position
        print("⏳ Waiting 5 seconds before closing position...")
        time.sleep(5)
        print()
        
        print("📝 Closing test position...")
        close_result = client.place_market_order(TEST_SYMBOL, is_buy=False, size=TEST_SIZE, reduce_only=True)
        
        if close_result.get("success"):
            print()
            print("✅ Position closed successfully!")
            print("   Test complete - system ready for live trading!")
        else:
            print()
            print("⚠️  Manual close may be needed")
            print(f"   Close {TEST_SIZE} {TEST_SYMBOL} position manually")
    else:
        print()
        print("=" * 80)
        print("❌ TRADE EXECUTION FAILED")
        print("=" * 80)
        print()
        print("Possible issues:")
        print("   1. API key doesn't have trading permissions")
        print("   2. Agent wallet not linked to main wallet")
        print("   3. Insufficient margin/collateral")
        print("   4. Symbol not available")
        print()
        print("Check Hyperliquid dashboard and API key settings")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
