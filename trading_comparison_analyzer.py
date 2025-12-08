#!/usr/bin/env python3
"""
Trading Comparison Analyzer
Compares naive paper trading vs realistic execution simulation
Shows why strategies fail in real trading despite good backtest results
"""

import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingComparisonAnalyzer:
    """
    Analyzes and compares different trading execution models
    """
    
    def __init__(self):
        self.naive_trades = []
        self.realistic_trades = []
        
    def load_naive_results(self, log_file: str):
        """Load results from naive paper trading"""
        # Parse the tradepexlogs file
        # This would parse the existing logs
        pass
    
    def analyze_execution_gap(self, naive_result: Dict, realistic_result: Dict) -> Dict:
        """
        Analyze the gap between naive and realistic execution
        
        Args:
            naive_result: Results from naive paper trading
            realistic_result: Results from realistic simulation
            
        Returns:
            Dict with analysis
        """
        gap_analysis = {}
        
        # Calculate PnL difference
        naive_pnl = naive_result.get('total_pnl', 0)
        realistic_pnl = realistic_result.get('total_pnl', 0)
        pnl_gap = naive_pnl - realistic_pnl
        
        # Calculate win rate difference
        naive_win_rate = naive_result.get('win_rate', 0)
        realistic_win_rate = realistic_result.get('win_rate', 0)
        win_rate_gap = naive_win_rate - realistic_win_rate
        
        # Execution cost breakdown
        avg_slippage = realistic_result.get('avg_slippage_bps', 0)
        total_fees = realistic_result.get('total_fees', 0)
        
        gap_analysis = {
            'pnl_gap_usd': pnl_gap,
            'pnl_gap_percent': (pnl_gap / abs(naive_pnl)) * 100 if naive_pnl != 0 else 0,
            'win_rate_gap': win_rate_gap,
            'execution_costs': {
                'avg_slippage_bps': avg_slippage,
                'total_fees_usd': total_fees,
                'fees_percent_of_capital': realistic_result.get('fees_pct_of_capital', 0)
            },
            'key_insight': self._generate_insight(pnl_gap, avg_slippage, total_fees)
        }
        
        return gap_analysis
    
    def _generate_insight(self, pnl_gap: float, avg_slippage: float, total_fees: float) -> str:
        """Generate human-readable insight"""
        if pnl_gap > 0:
            return (f"Naive model overestimates profit by ${pnl_gap:.2f}. "
                   f"Main causes: {avg_slippage:.1f} bps avg slippage + ${total_fees:.2f} in fees.")
        else:
            return "Realistic model shows better or similar performance."
    
    def generate_comparison_report(self, naive_result: Dict, realistic_result: Dict) -> str:
        """
        Generate a comprehensive comparison report
        
        Args:
            naive_result: Results from naive paper trading
            realistic_result: Results from realistic simulation
            
        Returns:
            Formatted report string
        """
        gap = self.analyze_execution_gap(naive_result, realistic_result)
        
        report = f"""
{'='*100}
TRADING EXECUTION COMPARISON REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

📊 PERFORMANCE COMPARISON
{'-'*100}
                          NAIVE PAPER TRADING    |    REALISTIC SIMULATION    |    GAP
{'-'*100}
Total PnL:               ${naive_result.get('total_pnl', 0):>12,.2f}  |  ${realistic_result.get('total_pnl', 0):>12,.2f}  |  ${gap['pnl_gap_usd']:>+10,.2f}
Return %:                {naive_result.get('total_return_pct', 0):>12.2f}% |  {realistic_result.get('total_return_pct', 0):>12.2f}% |  {gap['pnl_gap_percent']:>+10.2f}%
Win Rate:                {naive_result.get('win_rate', 0):>12.1f}% |  {realistic_result.get('win_rate', 0):>12.1f}% |  {gap['win_rate_gap']:>+10.1f}%
Total Trades:            {naive_result.get('total_trades', 0):>12}   |  {realistic_result.get('total_trades', 0):>12}   |
{'-'*100}

💰 EXECUTION COSTS (Only in Realistic Model)
{'-'*100}
Average Slippage:        N/A                  |  {gap['execution_costs']['avg_slippage_bps']:.2f} bps
Total Fees:              N/A                  |  ${gap['execution_costs']['total_fees_usd']:.2f}
Fees % of Capital:       N/A                  |  {gap['execution_costs']['fees_percent_of_capital']:.2f}%
{'-'*100}

🔍 KEY INSIGHTS
{'-'*100}
{gap['key_insight']}

WHY THE GAP EXISTS:
1. Bid-Ask Spread: Naive model assumes trades execute at mid-price, but real orders
   must cross the spread, paying either the ask (for buys) or receiving the bid (for sells).
   
2. Slippage: Large orders walk through multiple price levels in the order book,
   resulting in worse average fill prices than the top-of-book price.
   
3. Trading Fees: Exchanges charge fees (typically 0.02-0.05% for makers, 0.05-0.1% 
   for takers). These compound over many trades.
   
4. Market Impact: The naive model doesn't account for how your orders affect prices,
   especially in less liquid markets or with larger position sizes.

RECOMMENDATIONS:
"""
        
        if gap['pnl_gap_usd'] > 100:
            report += """
⚠️  CRITICAL: The gap between naive and realistic results is significant.

Action Items:
1. Reduce position sizes to minimize slippage impact
2. Use limit orders instead of market orders where possible (become a maker)
3. Trade more liquid assets with tighter spreads
4. Adjust profit targets to account for execution costs
5. Consider that strategies need higher gross returns to be net profitable

"""
        elif gap['pnl_gap_usd'] > 10:
            report += """
⚠️  MODERATE: There's a noticeable execution cost impact.

Action Items:
1. Monitor slippage on each trade
2. Consider using limit orders for entries
3. Factor in ~0.1-0.2% execution costs per round trip

"""
        else:
            report += """
✅ MINIMAL: Execution costs are manageable. Strategy appears robust.

"""
        
        report += f"""
{'='*100}
"""
        
        return report
    
    def create_detailed_trade_comparison(self, naive_trades: List[Dict], 
                                        realistic_trades: List[Dict]) -> List[Dict]:
        """
        Create detailed trade-by-trade comparison
        
        Returns list of dicts with side-by-side comparison
        """
        comparison_data = []
        
        # Match trades by strategy and symbol (assuming same trades were attempted)
        for i, (naive, realistic) in enumerate(zip(naive_trades, realistic_trades)):
            comparison_data.append({
                'trade_num': i + 1,
                'strategy': naive.get('strategy', 'N/A'),
                'symbol': naive.get('symbol', 'N/A'),
                'direction': naive.get('direction', 'N/A'),
                'naive_entry': naive.get('entry_price', 0),
                'realistic_entry': realistic.get('actual_entry', 0),
                'entry_slippage_bps': realistic.get('slippage_bps', 0),
                'naive_pnl': naive.get('pnl_usd', 0),
                'realistic_pnl': realistic.get('pnl_usd', 0),
                'pnl_diff': naive.get('pnl_usd', 0) - realistic.get('pnl_usd', 0),
                'fees': realistic.get('total_fees', 0)
            })
        
        return comparison_data


def demonstrate_problem():
    """
    Demonstrate the problem with a simple example
    """
    print("=" * 100)
    print("DEMONSTRATION: Why Paper Trading Results are Misleading")
    print("=" * 100)
    print()
    
    # Example scenario
    print("SCENARIO: VWAP Mean Reversion Strategy on ETH")
    print("-" * 100)
    print()
    
    print("📊 NAIVE PAPER TRADING (What tradepexv1.py does):")
    print("   1. Signal: BUY ETH at $3001.49 (below VWAP lower band)")
    print("   2. Assumption: Execute at EXACTLY $3001.49")
    print("   3. Target: $3024.02 (VWAP)")
    print("   4. If price hits target → Profit: ($3024.02 - $3001.49) / $3001.49 = +0.75%")
    print("   5. On $4500 position → $33.75 profit")
    print()
    
    print("💰 REALISTIC EXECUTION (What actually happens):")
    print("   1. Signal: BUY ETH at $3001.49")
    print("   2. Order Book Reality:")
    print("      - Best Bid: $3001.00")
    print("      - Best Ask: $3002.00 (you must pay the ask!)")
    print("      - Spread: $1.00 (0.033% or 3.3 bps)")
    print("   3. Market Buy Execution:")
    print("      - First $2000 fills at $3002.00")
    print("      - Next $2000 fills at $3002.50 (walking up the book)")
    print("      - Last $500 fills at $3003.00")
    print("      → Average Fill: $3002.44 (not $3001.49!)")
    print("   4. Entry Slippage: ($3002.44 - $3001.49) = $0.95 or 3.2 bps")
    print("   5. Entry Fee (0.05% taker): $2.25")
    print("   6. Net Entry Cost: $3002.44 + fees")
    print()
    print("   7. Exit at Target $3024.02:")
    print("      - Must sell at BID side (not mid)")
    print("      - Best Bid at target time: ~$3023.00")
    print("      - Exit Slippage: ~3 bps")
    print("      - Exit Fee (0.05% taker): $2.26")
    print()
    print("   8. Final Calculation:")
    print("      - Entry: $3002.44 + $2.25 fee")
    print("      - Exit: $3023.00 - $2.26 fee")
    print("      - Net Profit: ($3023.00 - $3002.44) - ($2.25 + $2.26) = $16.05")
    print("      - Net Return: +0.35% (not +0.75%!)")
    print()
    
    print("❌ THE PROBLEM:")
    print("   Naive Model: +0.75% return = $33.75 profit")
    print("   Reality:     +0.35% return = $16.05 profit")
    print("   LOSS:        -0.40% or      -$17.70 (52% reduction!)")
    print()
    
    print("🔄 MULTIPLY BY 100 TRADES:")
    print("   Naive Expectation: $3,375 profit")
    print("   Reality:           $1,605 profit")
    print("   GAP:               $1,770 (that's why strategies lose money!)")
    print()
    
    print("=" * 100)
    print("SOLUTION: Use realistic_trading_simulator.py which accounts for:")
    print("  ✓ Real order book spreads")
    print("  ✓ Slippage based on order size")
    print("  ✓ Trading fees (maker/taker)")
    print("  ✓ Market impact")
    print("=" * 100)


if __name__ == "__main__":
    demonstrate_problem()
    
    print("\n\n")
    
    # Example comparison report
    analyzer = TradingComparisonAnalyzer()
    
    naive_results = {
        'total_pnl': 850.50,
        'total_return_pct': 8.51,
        'win_rate': 65.0,
        'total_trades': 50,
        'avg_win': 25.50,
        'avg_loss': -18.20
    }
    
    realistic_results = {
        'total_pnl': 320.75,
        'total_return_pct': 3.21,
        'win_rate': 58.0,
        'total_trades': 50,
        'avg_win': 22.30,
        'avg_loss': -19.80,
        'avg_slippage_bps': 4.2,
        'total_fees': 125.50,
        'fees_pct_of_capital': 1.26
    }
    
    report = analyzer.generate_comparison_report(naive_results, realistic_results)
    print(report)
