#!/usr/bin/env python3
"""
Realistic Trading Simulator - Integrates Real Order Books with Trading Strategies
This replaces the naive paper trading with realistic execution simulation
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from hyperliquid_testnet import HyperliquidTestnetClient, ExecutionResult, OrderBook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RealisticPosition:
    """Tracks a position with realistic execution details"""
    
    def __init__(self, position_id: str, strategy_id: str, symbol: str, 
                 direction: str, intended_price: float, actual_entry_price: float,
                 size_usd: float, target: float, stop_loss: float,
                 execution_result: ExecutionResult):
        self.position_id = position_id
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.direction = direction
        self.intended_price = intended_price  # Price strategy wanted
        self.actual_entry_price = actual_entry_price  # Price we actually got
        self.size_usd = size_usd
        self.net_size = execution_result.filled_size  # After fees and slippage
        self.target_price = target
        self.stop_loss = stop_loss
        self.entry_time = datetime.now()
        self.status = "OPEN"
        self.exit_price = 0.0
        self.exit_execution: Optional[ExecutionResult] = None
        self.pnl = 0.0
        self.entry_slippage_bps = execution_result.slippage_bps
        self.entry_fees = execution_result.fees
        self.exit_slippage_bps = 0.0
        self.exit_fees = 0.0
        
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL"""
        if self.direction == "BUY":
            pnl_percent = (current_price - self.actual_entry_price) / self.actual_entry_price
        else:
            pnl_percent = (self.actual_entry_price - current_price) / self.actual_entry_price
        
        # PnL on the net position after entry fees
        unrealized = self.net_size * pnl_percent
        return unrealized
    
    def calculate_unrealized_pnl_percent(self, current_price: float) -> float:
        """Calculate unrealized PnL as percentage"""
        if self.actual_entry_price == 0:
            return 0.0
        return (self.calculate_unrealized_pnl(current_price) / self.size_usd) * 100


class RealisticTradingEngine:
    """
    Trading engine that simulates realistic execution using real order books
    """
    
    def __init__(self, starting_capital: float = 10000.0, use_hyperliquid: bool = True):
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.positions: Dict[str, RealisticPosition] = {}
        self.trade_history: List[Dict] = []
        self.execution_stats = {
            'total_slippage_bps': 0.0,
            'total_fees_usd': 0.0,
            'avg_entry_slippage_bps': 0.0,
            'avg_exit_slippage_bps': 0.0,
            'trade_count': 0
        }
        
        # Initialize market data client
        self.hl_client = HyperliquidTestnetClient(use_testnet=True) if use_hyperliquid else None
        
        # Fallback: HTX for historical data if needed
        self.htx_base_url = "https://api.huobi.pro"
        
        logger.info(f"🚀 Realistic Trading Engine initialized")
        logger.info(f"   Starting Capital: ${starting_capital:,.2f}")
        logger.info(f"   Using Hyperliquid: {use_hyperliquid}")
    
    def get_real_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Fetch real order book"""
        if self.hl_client:
            return self.hl_client.get_order_book(symbol)
        return None
    
    def open_position(self, strategy_id: str, symbol: str, signal: Dict,
                     size_percent: float = 0.15, max_slippage_bps: float = 50.0) -> Optional[str]:
        """
        Open position with realistic execution simulation
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading pair
            signal: Signal dict with 'signal', 'current_price', 'target_price', 'stop_loss'
            size_percent: Position size as % of capital
            max_slippage_bps: Maximum acceptable slippage
            
        Returns:
            position_id if successful, None otherwise
        """
        if signal['signal'] == 'HOLD':
            return None
        
        # Calculate position size
        size_usd = self.capital * size_percent
        leverage = 3  # Default leverage
        leveraged_size = size_usd * leverage
        
        # Get real order book
        order_book = self.get_real_order_book(symbol)
        if not order_book:
            logger.warning(f"❌ No order book available for {symbol}, skipping trade")
            return None
        
        # Log order book state
        logger.info(f"📊 Order Book - {symbol}")
        logger.info(f"   Best Bid: ${order_book.best_bid:,.2f}, Best Ask: ${order_book.best_ask:,.2f}")
        logger.info(f"   Spread: {order_book.spread_bps:.2f} bps")
        
        # Check available liquidity
        liquidity_info = self.hl_client.get_available_liquidity(
            symbol, signal['signal'], max_slippage_bps
        )
        
        if liquidity_info['available_usd'] < leveraged_size:
            logger.warning(f"⚠️  Insufficient liquidity for {symbol} {signal['signal']}")
            logger.warning(f"   Wanted: ${leveraged_size:,.2f}, Available: ${liquidity_info['available_usd']:,.2f}")
            # Reduce position size to available liquidity
            leveraged_size = liquidity_info['available_usd'] * 0.8  # Use 80% of available
            size_usd = leveraged_size / leverage
        
        # Simulate market order execution
        execution = self.hl_client.simulate_market_order(
            symbol, signal['signal'], leveraged_size, order_book
        )
        
        if not execution.success:
            logger.error(f"❌ Execution failed: {execution.error_message}")
            return None
        
        # Check if slippage is acceptable
        if execution.slippage_bps > max_slippage_bps:
            logger.warning(f"❌ Slippage too high: {execution.slippage_bps:.2f} bps > {max_slippage_bps} bps")
            return None
        
        # Create position
        position_id = f"{strategy_id}_{symbol}_{datetime.now().strftime('%H%M%S%f')}"
        
        position = RealisticPosition(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            direction=signal['signal'],
            intended_price=signal['current_price'],
            actual_entry_price=execution.avg_fill_price,
            size_usd=size_usd,
            target=signal.get('target_price', 0),
            stop_loss=signal.get('stop_loss', 0),
            execution_result=execution
        )
        
        self.positions[position_id] = position
        
        # Update stats
        self.execution_stats['total_slippage_bps'] += execution.slippage_bps
        self.execution_stats['total_fees_usd'] += execution.fees
        self.execution_stats['trade_count'] += 1
        self.execution_stats['avg_entry_slippage_bps'] = (
            self.execution_stats['total_slippage_bps'] / self.execution_stats['trade_count']
        )
        
        # Log trade
        trade_log = {
            'timestamp': datetime.now(),
            'position_id': position_id,
            'strategy': strategy_id,
            'symbol': symbol,
            'action': 'OPEN',
            'direction': signal['signal'],
            'intended_price': signal['current_price'],
            'actual_entry': execution.avg_fill_price,
            'size_usd': size_usd,
            'leveraged_size': leveraged_size,
            'target': signal.get('target_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'slippage_bps': execution.slippage_bps,
            'fees': execution.fees,
            'spread_bps': order_book.spread_bps,
            'reason': signal['reason']
        }
        
        self.trade_history.append(trade_log)
        
        logger.info(f"🎯 OPENED: {position_id}")
        logger.info(f"   {signal['signal']} {symbol}")
        logger.info(f"   Intended: ${signal['current_price']:.2f}, Actual: ${execution.avg_fill_price:.2f}")
        logger.info(f"   Slippage: {execution.slippage_bps:.2f} bps, Fees: ${execution.fees:.2f}")
        logger.info(f"   Size: ${size_usd:.2f} (${leveraged_size:.2f} leveraged)")
        logger.info(f"   Target: ${signal.get('target_price', 0):.2f}, Stop: ${signal.get('stop_loss', 0):.2f}")
        
        return position_id
    
    def check_exits(self, current_prices: Dict[str, float], order_books: Dict[str, OrderBook]):
        """
        Check if positions should be closed with realistic execution
        
        Args:
            current_prices: Dict of symbol -> current price
            order_books: Dict of symbol -> OrderBook
        """
        for position_id, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue
            
            symbol = position.symbol
            current_price = current_prices.get(symbol)
            order_book = order_books.get(symbol)
            
            if not current_price or not order_book:
                continue
            
            # Calculate current PnL
            unrealized_pnl_percent = position.calculate_unrealized_pnl_percent(current_price)
            
            should_close = False
            close_reason = ""
            
            # Check target hit
            if position.direction == "BUY" and current_price >= position.target_price:
                should_close = True
                close_reason = f"TARGET_HIT ({unrealized_pnl_percent:+.2f}%)"
            elif position.direction == "SELL" and current_price <= position.target_price:
                should_close = True
                close_reason = f"TARGET_HIT ({unrealized_pnl_percent:+.2f}%)"
            
            # Check stop loss
            elif position.direction == "BUY" and current_price <= position.stop_loss:
                should_close = True
                close_reason = f"STOP_LOSS ({unrealized_pnl_percent:+.2f}%)"
            elif position.direction == "SELL" and current_price >= position.stop_loss:
                should_close = True
                close_reason = f"STOP_LOSS ({unrealized_pnl_percent:+.2f}%)"
            
            if should_close:
                self.close_position(position_id, order_book, close_reason)
    
    def close_position(self, position_id: str, order_book: OrderBook, reason: str):
        """
        Close position with realistic execution simulation
        
        Args:
            position_id: Position to close
            order_book: Current order book
            reason: Reason for closing
        """
        position = self.positions[position_id]
        
        # Simulate closing the position (reverse direction)
        close_side = "SELL" if position.direction == "BUY" else "BUY"
        
        # Close with the net position size
        exit_execution = self.hl_client.simulate_market_order(
            position.symbol, close_side, position.net_size, order_book
        )
        
        if not exit_execution.success:
            logger.error(f"❌ Failed to close {position_id}: {exit_execution.error_message}")
            return
        
        # Calculate final PnL
        if position.direction == "BUY":
            # Bought at actual_entry_price, sold at exit price
            pnl_percent = (exit_execution.avg_fill_price - position.actual_entry_price) / position.actual_entry_price
        else:
            # Sold at actual_entry_price, bought back at exit price
            pnl_percent = (position.actual_entry_price - exit_execution.avg_fill_price) / position.actual_entry_price
        
        # Calculate PnL in USD
        gross_pnl = position.net_size * pnl_percent
        net_pnl = gross_pnl - exit_execution.fees  # Subtract exit fees
        
        # Update position
        position.status = "CLOSED"
        position.exit_price = exit_execution.avg_fill_price
        position.exit_execution = exit_execution
        position.pnl = net_pnl
        position.exit_slippage_bps = exit_execution.slippage_bps
        position.exit_fees = exit_execution.fees
        
        # Update capital
        self.capital += net_pnl
        
        # Update stats
        total_fees = position.entry_fees + position.exit_fees
        total_slippage = position.entry_slippage_bps + position.exit_slippage_bps
        
        self.execution_stats['total_fees_usd'] += exit_execution.fees
        self.execution_stats['total_slippage_bps'] += exit_execution.slippage_bps
        if position.status == "CLOSED":
            exit_count = len([p for p in self.positions.values() if p.status == "CLOSED"])
            if exit_count > 0:
                self.execution_stats['avg_exit_slippage_bps'] = sum(
                    p.exit_slippage_bps for p in self.positions.values() if p.status == "CLOSED"
                ) / exit_count
        
        # Log trade
        trade_log = {
            'timestamp': datetime.now(),
            'position_id': position_id,
            'strategy': position.strategy_id,
            'symbol': position.symbol,
            'action': 'CLOSE',
            'direction': position.direction,
            'entry_price': position.actual_entry_price,
            'exit_price': exit_execution.avg_fill_price,
            'intended_entry': position.intended_price,
            'pnl_usd': net_pnl,
            'gross_pnl_usd': gross_pnl,
            'pnl_percent': pnl_percent * 100,
            'entry_slippage_bps': position.entry_slippage_bps,
            'exit_slippage_bps': exit_execution.slippage_bps,
            'total_slippage_bps': total_slippage,
            'entry_fees': position.entry_fees,
            'exit_fees': exit_execution.fees,
            'total_fees': total_fees,
            'reason': reason
        }
        
        self.trade_history.append(trade_log)
        
        logger.info(f"🔒 CLOSED: {position_id}")
        logger.info(f"   {position.symbol} {position.direction}")
        logger.info(f"   Entry: ${position.actual_entry_price:.2f}, Exit: ${exit_execution.avg_fill_price:.2f}")
        logger.info(f"   Gross PnL: ${gross_pnl:+.2f}, Net PnL: ${net_pnl:+.2f} ({pnl_percent*100:+.2f}%)")
        logger.info(f"   Total Slippage: {total_slippage:.2f} bps, Total Fees: ${total_fees:.2f}")
        logger.info(f"   Reason: {reason}")
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        closed_positions = [p for p in self.positions.values() if p.status == "CLOSED"]
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        
        if not closed_positions:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'avg_slippage_bps': 0.0,
                'total_fees': 0.0
            }
        
        total_pnl = sum(p.pnl for p in closed_positions)
        winning_trades = [p for p in closed_positions if p.pnl > 0]
        losing_trades = [p for p in closed_positions if p.pnl <= 0]
        
        return {
            'total_trades': len(closed_positions),
            'open_positions': len(open_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_positions) * 100 if closed_positions else 0,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.starting_capital) * 100,
            'current_capital': self.capital,
            'avg_win': sum(p.pnl for p in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(p.pnl for p in losing_trades) / len(losing_trades) if losing_trades else 0,
            'avg_entry_slippage_bps': self.execution_stats['avg_entry_slippage_bps'],
            'avg_exit_slippage_bps': self.execution_stats['avg_exit_slippage_bps'],
            'total_fees': self.execution_stats['total_fees_usd'],
            'fees_pct_of_capital': (self.execution_stats['total_fees_usd'] / self.starting_capital) * 100
        }
    
    def display_status(self):
        """Display current status"""
        summary = self.get_performance_summary()
        
        print(f"\n{'='*100}")
        print(f"💰 REALISTIC TRADING ENGINE - EXECUTION ANALYSIS")
        print(f"{'='*100}")
        print(f"💵 Capital: ${self.capital:,.2f} | Return: {summary['total_return_pct']:+.2f}%")
        print(f"📊 Positions: {summary['open_positions']} open | {summary['total_trades']} closed")
        
        if summary['total_trades'] > 0:
            print(f"📈 Performance: {summary['winning_trades']}W / {summary['losing_trades']}L "
                  f"({summary['win_rate']:.1f}% win rate)")
            print(f"💰 Avg Win: ${summary['avg_win']:+.2f} | Avg Loss: ${summary['avg_loss']:+.2f}")
            print(f"⚡ Execution Costs:")
            print(f"   Entry Slippage: {summary['avg_entry_slippage_bps']:.2f} bps")
            print(f"   Exit Slippage: {summary['avg_exit_slippage_bps']:.2f} bps")
            print(f"   Total Fees: ${summary['total_fees']:.2f} ({summary['fees_pct_of_capital']:.2f}% of capital)")
        
        # Show open positions
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        if open_positions:
            print(f"\n📊 OPEN POSITIONS:")
            for position in open_positions[:10]:  # Show up to 10
                # Note: Would need current price to show unrealized PnL
                print(f"   {position.strategy_id[:30]:<30} {position.symbol:<6} {position.direction:<4} "
                      f"Entry: ${position.actual_entry_price:<10.2f} "
                      f"Target: ${position.target_price:<10.2f} "
                      f"Slippage: {position.entry_slippage_bps:.1f}bps")
        
        print(f"{'='*100}\n")


if __name__ == "__main__":
    # Simple test
    print("🧪 Testing Realistic Trading Engine\n")
    
    engine = RealisticTradingEngine(starting_capital=10000.0, use_hyperliquid=True)
    
    # Test signal
    test_signal = {
        'signal': 'BUY',
        'current_price': 90000.0,
        'target_price': 91000.0,
        'stop_loss': 89500.0,
        'reason': 'Test signal'
    }
    
    position_id = engine.open_position('test_strategy', 'BTC', test_signal, size_percent=0.1)
    
    if position_id:
        print(f"\n✅ Test position opened: {position_id}")
        engine.display_status()
    else:
        print("\n❌ Failed to open test position")
