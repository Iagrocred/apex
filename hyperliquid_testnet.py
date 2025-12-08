#!/usr/bin/env python3
"""
Real Order Book Integration for Trading Simulation
Uses Binance API to fetch real order book data for realistic trade execution simulation
"""

import os
import json
import time
import requests
import hmac
import hashlib
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book"""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book with bids and asks"""
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    timestamp: datetime
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points"""
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return (self.spread / self.best_bid) * 10000
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    success: bool
    filled_size: float
    avg_fill_price: float
    total_cost: float
    slippage_bps: float
    remaining_size: float
    fees: float
    error_message: Optional[str] = None


class HyperliquidTestnetClient:
    """
    Order Book Client for Real Trading Simulation
    Uses Binance API for order book data (publicly accessible)
    """
    
    # Use Binance API as it's publicly accessible and has good liquidity data
    BINANCE_API_URL = "https://api.binance.com"
    
    def __init__(self, use_testnet: bool = True, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.use_testnet = use_testnet
        self.api_url = self.BINANCE_API_URL
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
    def get_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        Fetch real order book from Binance (or simulated if network unavailable)
        
        Args:
            symbol: Trading pair (e.g., 'BTC', 'ETH')
            depth: Number of levels to fetch
            
        Returns:
            OrderBook object or None if failed
        """
        try:
            # Binance API endpoint for order book
            endpoint = f"{self.api_url}/api/v3/depth"
            
            # Convert symbol to Binance format (e.g., BTC -> BTCUSDT)
            binance_symbol = f"{symbol}USDT"
            
            params = {
                "symbol": binance_symbol,
                "limit": min(depth, 5000)  # Binance max is 5000
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse order book levels
                bids = []
                asks = []
                
                # Binance format: [[price, quantity], ...]
                for bid_data in data.get("bids", [])[:depth]:
                    bids.append(OrderBookLevel(
                        price=float(bid_data[0]),
                        size=float(bid_data[1])
                    ))
                
                for ask_data in data.get("asks", [])[:depth]:
                    asks.append(OrderBookLevel(
                        price=float(ask_data[0]),
                        size=float(ask_data[1])
                    ))
                
                return OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now()
                )
            else:
                logger.error(f"Failed to fetch order book: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Network unavailable for {symbol}, using simulated order book: {e}")
            return self._generate_simulated_order_book(symbol, depth)
    
    def _generate_simulated_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        """
        Generate a realistic simulated order book based on typical market conditions
        Uses historical spread and liquidity patterns
        """
        # Get mid price from HTX or use default
        mid_price = self._get_fallback_price(symbol)
        if not mid_price:
            logger.error(f"Cannot generate simulated order book without price for {symbol}")
            return None
        
        # Typical spreads in basis points for different assets
        spread_bps_map = {
            'BTC': 1.0,   # ~0.01% spread
            'ETH': 1.5,   # ~0.015% spread
            'SOL': 3.0,   # ~0.03% spread
            'XRP': 4.0,
            'ADA': 5.0,
            'DOT': 4.0,
            'LINK': 5.0,
            'AVAX': 4.0
        }
        
        spread_bps = spread_bps_map.get(symbol, 5.0)
        spread = mid_price * (spread_bps / 10000)
        
        best_bid = mid_price - (spread / 2)
        best_ask = mid_price + (spread / 2)
        
        # Generate order book levels with realistic size distribution
        # Liquidity decreases as we move away from mid price
        bids = []
        asks = []
        
        # Typical size at best level (in USD equivalent)
        base_size_usd = {
            'BTC': 500000,  # $500k at best level
            'ETH': 300000,
            'SOL': 100000,
            'XRP': 50000,
            'ADA': 30000,
            'DOT': 40000,
            'LINK': 50000,
            'AVAX': 40000
        }
        
        base_size = base_size_usd.get(symbol, 50000) / mid_price
        
        for i in range(depth):
            # Price moves away from best by small increments
            price_step = spread * 0.2  # 20% of spread per level
            
            # Bid side
            bid_price = best_bid - (i * price_step)
            # Size increases slightly with depth (iceberg orders, hidden liquidity)
            bid_size = base_size * (1 + i * 0.1) * (0.8 + 0.4 * random.random())
            bids.append(OrderBookLevel(price=bid_price, size=bid_size))
            
            # Ask side
            ask_price = best_ask + (i * price_step)
            ask_size = base_size * (1 + i * 0.1) * (0.8 + 0.4 * random.random())
            asks.append(OrderBookLevel(price=ask_price, size=ask_size))
        
        logger.info(f"📊 Generated simulated order book for {symbol}")
        logger.info(f"   Mid: ${mid_price:.2f}, Spread: {spread_bps:.2f} bps")
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
    
    def _get_fallback_price(self, symbol: str) -> Optional[float]:
        """Get price from HTX as fallback"""
        try:
            htx_url = "https://api.huobi.pro"
            endpoint = f"{htx_url}/market/history/kline"
            params = {
                "symbol": f"{symbol.lower()}usdt",
                "period": "1min",
                "size": 1
            }
            response = self.session.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("data"):
                    return float(data["data"][0]["close"])
        except Exception as e:
            logger.warning(f"HTX fallback failed for {symbol}: {e}")
        
        # Last resort: use approximate prices
        fallback_prices = {
            'BTC': 95000.0,
            'ETH': 3500.0,
            'SOL': 200.0,
            'XRP': 2.5,
            'ADA': 1.0,
            'DOT': 8.0,
            'LINK': 20.0,
            'AVAX': 40.0
        }
        return fallback_prices.get(symbol)
    
    def simulate_market_order(self, symbol: str, side: str, size_usd: float, 
                             order_book: Optional[OrderBook] = None) -> ExecutionResult:
        """
        Simulate market order execution using real order book
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            size_usd: Order size in USD
            order_book: Optional pre-fetched order book
            
        Returns:
            ExecutionResult with fill details
        """
        if order_book is None:
            order_book = self.get_order_book(symbol)
        
        if order_book is None:
            return ExecutionResult(
                success=False,
                filled_size=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                slippage_bps=0.0,
                remaining_size=size_usd,
                fees=0.0,
                error_message="Failed to fetch order book"
            )
        
        # Determine which side of the book to use
        levels = order_book.asks if side == 'BUY' else order_book.bids
        reference_price = order_book.best_ask if side == 'BUY' else order_book.best_bid
        
        if not levels or reference_price is None:
            return ExecutionResult(
                success=False,
                filled_size=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                slippage_bps=0.0,
                remaining_size=size_usd,
                fees=0.0,
                error_message="Order book has no liquidity"
            )
        
        # Simulate filling through the order book
        remaining_usd = size_usd
        total_filled_qty = 0.0
        total_cost = 0.0
        fills = []
        
        for level in levels:
            if remaining_usd <= 0:
                break
            
            # Calculate how much we can fill at this level
            level_value_usd = level.size * level.price
            fill_usd = min(remaining_usd, level_value_usd)
            fill_qty = fill_usd / level.price
            
            total_filled_qty += fill_qty
            total_cost += fill_usd
            remaining_usd -= fill_usd
            fills.append((level.price, fill_qty))
        
        if total_filled_qty == 0:
            return ExecutionResult(
                success=False,
                filled_size=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                slippage_bps=0.0,
                remaining_size=size_usd,
                fees=0.0,
                error_message="No fills executed"
            )
        
        # Calculate average fill price
        avg_fill_price = total_cost / total_filled_qty
        
        # Calculate slippage in basis points
        slippage_bps = ((avg_fill_price - reference_price) / reference_price) * 10000
        if side == 'SELL':
            slippage_bps = -slippage_bps  # Slippage is negative for sells going down
        
        # Calculate fees (Hyperliquid testnet: ~0.02% maker, ~0.05% taker)
        # Market orders are always taker
        taker_fee_rate = 0.0005  # 0.05%
        fees = total_cost * taker_fee_rate
        
        return ExecutionResult(
            success=True,
            filled_size=total_cost - fees,
            avg_fill_price=avg_fill_price,
            total_cost=total_cost + fees,
            slippage_bps=abs(slippage_bps),
            remaining_size=remaining_usd,
            fees=fees
        )
    
    def simulate_limit_order(self, symbol: str, side: str, size_usd: float, 
                           limit_price: float, order_book: Optional[OrderBook] = None) -> ExecutionResult:
        """
        Simulate limit order - checks if it would fill immediately or rest on book
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            size_usd: Order size in USD
            limit_price: Limit price
            order_book: Optional pre-fetched order book
            
        Returns:
            ExecutionResult with fill details
        """
        if order_book is None:
            order_book = self.get_order_book(symbol)
        
        if order_book is None:
            return ExecutionResult(
                success=False,
                filled_size=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                slippage_bps=0.0,
                remaining_size=size_usd,
                fees=0.0,
                error_message="Failed to fetch order book"
            )
        
        # Check if limit order would cross the spread (immediate fill)
        if side == 'BUY':
            # Buy limit crosses if limit >= best ask
            if order_book.best_ask and limit_price >= order_book.best_ask:
                # Would fill as market order up to limit price
                return self._simulate_limit_market_fill(
                    symbol, side, size_usd, limit_price, order_book
                )
            else:
                # Would rest on book as maker - calculate potential fill
                maker_fee_rate = 0.0002  # 0.02% maker fee
                fill_qty = size_usd / limit_price
                fees = size_usd * maker_fee_rate
                
                return ExecutionResult(
                    success=True,
                    filled_size=size_usd - fees,
                    avg_fill_price=limit_price,
                    total_cost=size_usd + fees,
                    slippage_bps=0.0,  # No slippage for maker orders
                    remaining_size=0.0,
                    fees=fees
                )
        else:  # SELL
            # Sell limit crosses if limit <= best bid
            if order_book.best_bid and limit_price <= order_book.best_bid:
                return self._simulate_limit_market_fill(
                    symbol, side, size_usd, limit_price, order_book
                )
            else:
                maker_fee_rate = 0.0002
                fill_qty = size_usd / limit_price
                fees = size_usd * maker_fee_rate
                
                return ExecutionResult(
                    success=True,
                    filled_size=size_usd - fees,
                    avg_fill_price=limit_price,
                    total_cost=size_usd - fees,  # Selling, so we receive
                    slippage_bps=0.0,
                    remaining_size=0.0,
                    fees=fees
                )
    
    def _simulate_limit_market_fill(self, symbol: str, side: str, size_usd: float,
                                   limit_price: float, order_book: OrderBook) -> ExecutionResult:
        """Simulate limit order that crosses spread and fills as taker"""
        levels = order_book.asks if side == 'BUY' else order_book.bids
        reference_price = order_book.best_ask if side == 'BUY' else order_book.best_bid
        
        remaining_usd = size_usd
        total_filled_qty = 0.0
        total_cost = 0.0
        
        for level in levels:
            # Stop if we've hit our limit price
            if side == 'BUY' and level.price > limit_price:
                break
            if side == 'SELL' and level.price < limit_price:
                break
            
            if remaining_usd <= 0:
                break
            
            level_value_usd = level.size * level.price
            fill_usd = min(remaining_usd, level_value_usd)
            fill_qty = fill_usd / level.price
            
            total_filled_qty += fill_qty
            total_cost += fill_usd
            remaining_usd -= fill_usd
        
        if total_filled_qty == 0:
            return ExecutionResult(
                success=False,
                filled_size=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                slippage_bps=0.0,
                remaining_size=size_usd,
                fees=0.0,
                error_message="No fills at limit price"
            )
        
        avg_fill_price = total_cost / total_filled_qty
        slippage_bps = ((avg_fill_price - reference_price) / reference_price) * 10000
        if side == 'SELL':
            slippage_bps = -slippage_bps
        
        taker_fee_rate = 0.0005
        fees = total_cost * taker_fee_rate
        
        return ExecutionResult(
            success=True,
            filled_size=total_cost - fees,
            avg_fill_price=avg_fill_price,
            total_cost=total_cost + fees,
            slippage_bps=abs(slippage_bps),
            remaining_size=remaining_usd,
            fees=fees
        )
    
    def get_available_liquidity(self, symbol: str, side: str, 
                               max_slippage_bps: float = 100.0) -> Dict:
        """
        Calculate available liquidity within a slippage tolerance
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            max_slippage_bps: Maximum acceptable slippage in basis points
            
        Returns:
            Dict with liquidity info
        """
        order_book = self.get_order_book(symbol)
        if not order_book:
            return {
                'available_usd': 0.0,
                'levels_count': 0,
                'avg_price': 0.0,
                'error': 'Failed to fetch order book'
            }
        
        levels = order_book.asks if side == 'BUY' else order_book.bids
        reference_price = order_book.best_ask if side == 'BUY' else order_book.best_bid
        
        if not reference_price:
            return {
                'available_usd': 0.0,
                'levels_count': 0,
                'avg_price': 0.0,
                'error': 'No reference price'
            }
        
        max_price_deviation = reference_price * (max_slippage_bps / 10000)
        max_price = reference_price + max_price_deviation if side == 'BUY' else reference_price - max_price_deviation
        
        total_liquidity_usd = 0.0
        levels_count = 0
        
        for level in levels:
            if side == 'BUY' and level.price > max_price:
                break
            if side == 'SELL' and level.price < max_price:
                break
            
            total_liquidity_usd += level.size * level.price
            levels_count += 1
        
        return {
            'available_usd': total_liquidity_usd,
            'levels_count': levels_count,
            'avg_price': max_price,
            'reference_price': reference_price,
            'max_slippage_bps': max_slippage_bps,
            'spread_bps': order_book.spread_bps
        }


def test_hyperliquid_integration():
    """Test the Order Book integration"""
    print("🧪 Testing Order Book Integration (Binance API)\n")
    
    client = HyperliquidTestnetClient(use_testnet=True)
    
    # Test 1: Fetch order book
    print("📊 Test 1: Fetching BTC order book...")
    btc_book = client.get_order_book('BTC')
    if btc_book:
        print(f"✅ Order book fetched successfully")
        print(f"   Best Bid: ${btc_book.best_bid:,.2f}")
        print(f"   Best Ask: ${btc_book.best_ask:,.2f}")
        print(f"   Spread: ${btc_book.spread:.2f} ({btc_book.spread_bps:.1f} bps)")
        print(f"   Bid Levels: {len(btc_book.bids)}, Ask Levels: {len(btc_book.asks)}")
    else:
        print("❌ Failed to fetch order book")
        return
    
    # Test 2: Simulate small market buy
    print("\n📊 Test 2: Simulating $1000 market BUY...")
    result = client.simulate_market_order('BTC', 'BUY', 1000.0, btc_book)
    if result.success:
        print(f"✅ Order simulated successfully")
        print(f"   Avg Fill Price: ${result.avg_fill_price:,.2f}")
        print(f"   Slippage: {result.slippage_bps:.2f} bps")
        print(f"   Fees: ${result.fees:.2f}")
        print(f"   Net Filled: ${result.filled_size:.2f}")
    else:
        print(f"❌ Simulation failed: {result.error_message}")
    
    # Test 3: Simulate larger market buy to see slippage
    print("\n📊 Test 3: Simulating $10,000 market BUY...")
    result = client.simulate_market_order('BTC', 'BUY', 10000.0, btc_book)
    if result.success:
        print(f"✅ Order simulated successfully")
        print(f"   Avg Fill Price: ${result.avg_fill_price:,.2f}")
        print(f"   Slippage: {result.slippage_bps:.2f} bps")
        print(f"   Fees: ${result.fees:.2f}")
        print(f"   Net Filled: ${result.filled_size:.2f}")
    
    # Test 4: Check available liquidity
    print("\n📊 Test 4: Checking available liquidity (max 50 bps slippage)...")
    liquidity = client.get_available_liquidity('BTC', 'BUY', max_slippage_bps=50.0)
    print(f"   Available Liquidity: ${liquidity['available_usd']:,.2f}")
    print(f"   Levels: {liquidity['levels_count']}")
    print(f"   Current Spread: {liquidity.get('spread_bps', 0):.2f} bps")


if __name__ == "__main__":
    test_hyperliquid_integration()
