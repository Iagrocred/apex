#!/usr/bin/env python3
"""
TRADEPEX v2 - REALISTIC EXECUTION ENGINE
Now with proper order book simulation, slippage, and fee modeling
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging

# Import our new realistic trading components
from hyperliquid_testnet import HyperliquidTestnetClient
from realistic_trading_simulator import RealisticTradingEngine, RealisticPosition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Position check result type
class PositionCheck(NamedTuple):
    """Result of position opening eligibility check"""
    allowed: bool
    reason: str


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    STARTING_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.15  # 15% per trade
    DEFAULT_LEVERAGE = 3
    TRADEABLE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'AVAX']
    STRATEGIES_DIR = Path("/root/KEEP_SAFE/v1/APEX/successful_strategies")
    CHECK_INTERVAL = 60  # Check every minute
    
    # Position limits
    MAX_TOTAL_POSITIONS = 8
    MAX_POSITIONS_PER_STRATEGY = 2
    MAX_POSITIONS_PER_TOKEN = 2
    
    # Execution parameters
    MAX_SLIPPAGE_BPS = 50.0  # Max 50 bps slippage per trade
    USE_REALISTIC_EXECUTION = True  # Toggle for comparison


# =============================================================================
# REAL HTX CLIENT FOR PRICE DATA
# =============================================================================

class RealHTXClient:
    def __init__(self):
        self.base_url = "https://api.huobi.pro"
        # Note: Will use fallback prices if network unavailable
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price with fallback"""
        # In real scenario, this would fetch from HTX
        # For now, use fallback prices
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
        return fallback_prices.get(symbol, 100.0)


# =============================================================================
# STRATEGY EXECUTION ENGINE
# =============================================================================

class StrategyExecutor:
    def calculate_vwap_signal(self, symbol: str, current_price: float) -> dict:
        """
        Simple VWAP-based signal generation
        In production, this would use real historical data
        """
        # Simulate VWAP bands (in production, calculate from historical data)
        vwap = current_price
        std_dev = current_price * 0.01  # 1% volatility assumption
        
        upper_band = vwap + (2.0 * std_dev)
        lower_band = vwap - (2.0 * std_dev)
        
        atr = current_price * 0.015  # 1.5% ATR assumption
        
        signal = "HOLD"
        reason = "Price within bands"
        target_price = 0.0
        stop_loss = 0.0
        
        # Check for signals
        lower_deviation = (lower_band - current_price) / current_price * 100
        upper_deviation = (current_price - upper_band) / current_price * 100
        
        if current_price < lower_band and abs(lower_deviation) > 0.3:
            signal = "BUY"
            reason = f"Price ${current_price:.2f} below VWAP lower band ${lower_band:.2f}"
            target_price = vwap
            stop_loss = current_price - (atr * 1.5)
        
        elif current_price > upper_band and abs(upper_deviation) > 0.3:
            signal = "SELL"
            reason = f"Price ${current_price:.2f} above VWAP upper band ${upper_band:.2f}"
            target_price = vwap
            stop_loss = current_price + (atr * 1.5)
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'vwap': vwap,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'atr': atr
        }


# =============================================================================
# MAIN TRADING ENGINE
# =============================================================================

class LiveTradingEngineV2:
    def __init__(self):
        self.htx_client = RealHTXClient()
        self.strategy_executor = StrategyExecutor()
        
        # Use realistic trading engine
        self.trading_engine = RealisticTradingEngine(
            starting_capital=Config.STARTING_CAPITAL,
            use_hyperliquid=Config.USE_REALISTIC_EXECUTION
        )
        
        self.strategies = self.load_strategies()
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        logger.info("🚀 TRADEPEX V2 - REALISTIC EXECUTION ENGINE")
        logger.info(f"   Starting Capital: ${Config.STARTING_CAPITAL:,.2f}")
        logger.info(f"   Realistic Execution: {Config.USE_REALISTIC_EXECUTION}")
        logger.info(f"   Max Slippage: {Config.MAX_SLIPPAGE_BPS} bps")
        
    def load_strategies(self) -> dict:
        """Load strategy metadata"""
        strategies = {}
        
        # Simplified: Use predefined strategy types
        strategy_types = [
            'VWAP_Mean_Reversion',
            'Market_Maker',
            'Pairs_Trading'
        ]
        
        for i, strategy_type in enumerate(strategy_types):
            strategy_id = f"Strategy_{i+1}_{strategy_type}"
            strategies[strategy_id] = {
                'type': strategy_type,
                'name': strategy_id
            }
        
        logger.info(f"📚 Loaded {len(strategies)} strategies")
        return strategies
    
    def can_open_position(self, strategy_id: str, symbol: str) -> PositionCheck:
        """Check if we can open a new position
        
        Returns:
            PositionCheck with allowed flag and reason
        """
        open_positions = [p for p in self.trading_engine.positions.values() if p.status == "OPEN"]
        
        # Check total limit
        if len(open_positions) >= Config.MAX_TOTAL_POSITIONS:
            return PositionCheck(False, f"Max total positions ({Config.MAX_TOTAL_POSITIONS})")
        
        # Check strategy limit
        strategy_positions = [p for p in open_positions if p.strategy_id == strategy_id]
        if len(strategy_positions) >= Config.MAX_POSITIONS_PER_STRATEGY:
            return PositionCheck(False, f"Max positions for {strategy_id}")
        
        # Check token limit
        token_positions = [p for p in open_positions if p.symbol == symbol]
        if len(token_positions) >= Config.MAX_POSITIONS_PER_TOKEN:
            return PositionCheck(False, f"Max positions for {symbol}")
        
        # Check if already have this exact position
        existing = [p for p in open_positions 
                   if p.strategy_id == strategy_id and p.symbol == symbol]
        if existing:
            return PositionCheck(False, f"Already have position")
        
        return PositionCheck(True, "OK")
    
    def execute_strategy(self, strategy_id: str, symbol: str):
        """Execute strategy for a symbol"""
        try:
            # Get current price
            current_price = self.htx_client.get_current_price(symbol)
            if not current_price:
                return
            
            # Generate signal
            signal = self.strategy_executor.calculate_vwap_signal(symbol, current_price)
            
            if signal['signal'] == 'HOLD':
                return
            
            # Check if we can open position
            check = self.can_open_position(strategy_id, symbol)
            if not check.allowed:
                logger.info(f"⏸️  {strategy_id} {symbol} - {check.reason}")
                return
            
            # Open position with realistic execution
            logger.info(f"🚀 SIGNAL: {strategy_id} {signal['signal']} {symbol}")
            logger.info(f"   {signal['reason']}")
            
            position_id = self.trading_engine.open_position(
                strategy_id=strategy_id,
                symbol=symbol,
                signal=signal,
                size_percent=Config.MAX_POSITION_SIZE,
                max_slippage_bps=Config.MAX_SLIPPAGE_BPS
            )
            
            if position_id:
                logger.info(f"✅ Position opened: {position_id}")
            
        except Exception as e:
            logger.error(f"❌ Error executing {strategy_id} for {symbol}: {e}")
    
    def check_exits(self):
        """Check if any positions should be closed"""
        # Get current prices and order books for all open positions
        open_positions = [p for p in self.trading_engine.positions.values() 
                         if p.status == "OPEN"]
        
        if not open_positions:
            return
        
        # Get unique symbols
        symbols = list(set(p.symbol for p in open_positions))
        
        current_prices = {}
        order_books = {}
        
        for symbol in symbols:
            price = self.htx_client.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
                # Get order book
                order_book = self.trading_engine.hl_client.get_order_book(symbol)
                if order_book:
                    order_books[symbol] = order_book
        
        # Check exits with realistic execution
        self.trading_engine.check_exits(current_prices, order_books)
    
    def display_status(self):
        """Display current status"""
        summary = self.trading_engine.get_performance_summary()
        
        print(f"\n{'='*100}")
        print(f"🔄 TRADEPEX V2 - REALISTIC EXECUTION ENGINE")
        print(f"{'='*100}")
        print(f"⏰ Cycle: {self.cycle_count} | Runtime: {datetime.now() - self.start_time}")
        print(f"💰 Capital: ${self.trading_engine.capital:,.2f} | "
              f"Return: {summary['total_return_pct']:+.2f}%")
        print(f"📊 Positions: {summary['open_positions']} open | "
              f"{summary['total_trades']} closed")
        
        if summary['total_trades'] > 0:
            print(f"📈 Win Rate: {summary['win_rate']:.1f}% "
                  f"({summary['winning_trades']}W / {summary['losing_trades']}L)")
            print(f"⚡ Execution Costs:")
            print(f"   Avg Entry Slippage: {summary['avg_entry_slippage_bps']:.2f} bps")
            print(f"   Avg Exit Slippage: {summary['avg_exit_slippage_bps']:.2f} bps")
            print(f"   Total Fees: ${summary['total_fees']:.2f} "
                  f"({summary['fees_pct_of_capital']:.2f}% of capital)")
        
        # Show open positions
        open_positions = [p for p in self.trading_engine.positions.values() 
                         if p.status == "OPEN"]
        if open_positions:
            print(f"\n📊 OPEN POSITIONS:")
            for position in open_positions[:5]:  # Show first 5
                current_price = self.htx_client.get_current_price(position.symbol)
                if current_price:
                    unrealized_pnl_pct = position.calculate_unrealized_pnl_percent(current_price)
                    print(f"   {position.strategy_id[:25]:<25} {position.symbol:<6} "
                          f"{position.direction:<4} Entry: ${position.actual_entry_price:<10.2f} "
                          f"Current: ${current_price:<10.2f} PnL: {unrealized_pnl_pct:+.2f}%")
        
        print(f"{'='*100}\n")
    
    def run_loop(self, max_cycles: int = None):
        """Run the trading loop"""
        logger.info("🚀 Starting TRADEPEX V2 trading loop...")
        logger.info(f"   Strategies: {len(self.strategies)}")
        logger.info(f"   Tokens: {Config.TRADEABLE_TOKENS}")
        logger.info(f"   Check interval: {Config.CHECK_INTERVAL}s")
        
        try:
            while True:
                self.cycle_count += 1
                
                if max_cycles and self.cycle_count > max_cycles:
                    logger.info(f"✅ Reached max cycles: {max_cycles}")
                    break
                
                logger.info(f"\n🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check exits first
                self.check_exits()
                
                # Look for new opportunities
                open_count = len([p for p in self.trading_engine.positions.values() 
                                 if p.status == "OPEN"])
                
                if open_count < Config.MAX_TOTAL_POSITIONS:
                    for strategy_id in self.strategies:
                        for token in Config.TRADEABLE_TOKENS:
                            self.execute_strategy(strategy_id, token)
                            time.sleep(0.5)  # Rate limiting
                else:
                    logger.info(f"⏸️  Position limit reached ({open_count}/{Config.MAX_TOTAL_POSITIONS})")
                
                # Display status
                self.display_status()
                
                # Sleep until next cycle
                if not max_cycles:  # Only sleep in continuous mode
                    time.sleep(Config.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested...")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final status
            logger.info("\n📊 FINAL STATUS:")
            self.display_status()
            
            # Save results
            self.save_results()
    
    def save_results(self):
        """Save trading results to file"""
        results = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'cycles': self.cycle_count,
            'summary': self.trading_engine.get_performance_summary(),
            'trades': self.trading_engine.trade_history
        }
        
        filename = f"tradepex_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filename}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("🎯 TRADEPEX V2 - REALISTIC EXECUTION ENGINE")
    print("   Now with proper order book simulation!")
    print("=" * 100)
    print()
    
    engine = LiveTradingEngineV2()
    
    # Run for a few cycles as demo
    engine.run_loop(max_cycles=3)
