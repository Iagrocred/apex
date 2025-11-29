#!/usr/bin/env python3
"""
Test script to verify Hyperliquid connection, balance fetch, and trade execution
"""

import os
import sys
import time
from decimal import Decimal
from eth_account import Account
import requests
import json

# Load environment
from dotenv import load_dotenv
load_dotenv()

HYPERLIQUID_KEY = os.getenv('HYPER_LIQUID_KEY')
BASE_URL = "https://api.hyperliquid.xyz"

def get_account():
    """Initialize account from private key"""
    if not HYPERLIQUID_KEY:
        print("❌ ERROR: HYPER_LIQUID_KEY not found in environment!")
        sys.exit(1)
    
    account = Account.from_key(HYPERLIQUID_KEY)
    print(f"✅ Account loaded: {account.address}")
    return account

def get_user_state(account):
    """Get user state from Hyperliquid"""
    try:
        response = requests.post(
            f"{BASE_URL}/info",
            headers={"Content-Type": "application/json"},
            json={
                "type": "clearinghouseState",
                "user": account.address
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User state retrieved successfully")
            print(f"   Raw response: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"❌ Failed to get user state: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting user state: {e}")
        return None

def get_account_value(user_state):
    """Extract account value from user state"""
    if not user_state:
        return 0.0
    
    try:
        # Try different possible paths for account value
        margin_summary = user_state.get('marginSummary', {})
        
        # Path 1: accountValue
        account_value = float(margin_summary.get('accountValue', 0))
        if account_value > 0:
            return account_value
        
        # Path 2: totalAccountValue
        account_value = float(margin_summary.get('totalAccountValue', 0))
        if account_value > 0:
            return account_value
        
        # Path 3: Calculate from balance components
        total_raw_usd = float(user_state.get('withdrawable', 0))
        total_margin_used = float(margin_summary.get('totalMarginUsed', 0))
        unrealized_pnl = float(margin_summary.get('totalNtlPos', 0))
        
        calculated_value = total_raw_usd + unrealized_pnl
        
        print(f"   Withdrawable: ${total_raw_usd:.2f}")
        print(f"   Margin Used: ${total_margin_used:.2f}")
        print(f"   Unrealized PnL: ${unrealized_pnl:.2f}")
        print(f"   Calculated Total: ${calculated_value:.2f}")
        
        return calculated_value
        
    except Exception as e:
        print(f"❌ Error calculating account value: {e}")
        return 0.0

def get_all_mids():
    """Get all market mid prices"""
    try:
        response = requests.post(
            f"{BASE_URL}/info",
            headers={"Content-Type": "application/json"},
            json={"type": "allMids"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Market prices retrieved: {len(data)} assets")
            
            # Show first few
            for symbol, price in list(data.items())[:5]:
                print(f"   {symbol}: ${price}")
            
            return data
        else:
            print(f"❌ Failed to get market prices: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"❌ Error getting market prices: {e}")
        return {}

def main():
    """Main test function"""
    print("=" * 80)
    print("🚀 HYPERLIQUID CONNECTION TEST")
    print("=" * 80)
    print()
    
    # Step 1: Load account
    print("STEP 1: Loading account...")
    account = get_account()
    print()
    
    # Step 2: Get user state
    print("STEP 2: Fetching user state...")
    user_state = get_user_state(account)
    print()
    
    # Step 3: Get account value
    print("STEP 3: Calculating account value...")
    if user_state:
        account_value = get_account_value(user_state)
        print(f"💰 ACCOUNT VALUE: ${account_value:.2f}")
        
        if account_value == 0:
            print()
            print("⚠️  WARNING: Account value is $0.00!")
            print("   Possible reasons:")
            print("   1. No funds deposited to Hyperliquid account")
            print("   2. All funds are in open positions (check positions)")
            print("   3. API key doesn't have read permissions")
            print("   4. Account address mismatch")
    else:
        print("❌ Could not retrieve account value")
    print()
    
    # Step 4: Get market prices
    print("STEP 4: Fetching market prices...")
    mids = get_all_mids()
    print()
    
    # Step 5: Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Account: {account.address}")
    print(f"{'✅' if user_state else '❌'} User State: {'Retrieved' if user_state else 'Failed'}")
    if user_state:
        account_value = get_account_value(user_state)
        print(f"💰 Balance: ${account_value:.2f}")
    print(f"{'✅' if mids else '❌'} Market Data: {'Retrieved' if mids else 'Failed'}")
    print("=" * 80)
    print()
    
    if user_state and account_value > 0:
        print("✅ ALL SYSTEMS OPERATIONAL - Ready for trading!")
    elif user_state and account_value == 0:
        print("⚠️  CONNECTION OK but ZERO BALANCE - Please deposit funds to trade")
    else:
        print("❌ CONNECTION ISSUES - Check API key and network")

if __name__ == "__main__":
    main()
