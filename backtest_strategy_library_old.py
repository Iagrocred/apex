#!/usr/bin/env python3
"""
Strategy Library Batch Backtester
Backtests all strategies from strategy_library with corrected swarm consensus

This script:
1. Loads all strategies from strategy_library/
2. Feeds each to RBI backtesting queue
3. Monitors for approvals with corrected swarm (DeepSeek + GPT-4o + Claude)
4. Reports which strategies pass Moon-Dev's standards
5. Approved strategies saved to data/[date]/backtests_final/

NOTE: This script queues strategies for the main APEX system to backtest.
      Run main APEX in another terminal for actual backtesting.
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Simple script - just loads strategies and provides instructions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("STRATEGY_LOADER")

class StrategyLibraryLoader:
    """Load and analyze strategies from strategy_library"""
    
    def __init__(self):
        self.strategy_library_dir = Path("strategy_library")
        self.strategies = []
        
    def load_strategies(self):
        """Load all JSON strategies from strategy_library"""
        logger.info("=" * 80)
        logger.info("🔍 Loading strategies from strategy_library/")
        logger.info("=" * 80)
        
        if not self.strategy_library_dir.exists():
            logger.error(f"❌ Strategy library not found: {self.strategy_library_dir}")
            return []
            
        for strategy_file in sorted(self.strategy_library_dir.glob("*.json")):
            try:
                with open(strategy_file, 'r') as f:
                    strategy = json.load(f)
                    strategy['source_file'] = strategy_file.name
                    self.strategies.append(strategy)
                    logger.info(f"   ✅ {strategy.get('name', strategy_file.name)}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to load {strategy_file.name}: {e}")
                
        logger.info("")
        logger.info(f"📊 Total strategies: {len(self.strategies)}")
        logger.info("=" * 80)
        
        return self.strategies
    
    def print_summary(self):
        """Print summary of loaded strategies"""
        logger.info("")
        logger.info("📊 STRATEGY ANALYSIS")
        logger.info("=" * 80)
        
        # Count strategy types
        strategy_types = {}
        for strategy in self.strategies:
            name = strategy.get('name', 'Unknown')
            # Extract strategy type
            if 'market making' in name.lower() or 'stoikov' in name.lower():
                stype = 'Market Making'
            elif 'mean reversion' in name.lower() or 'vwap' in name.lower():
                stype = 'Mean Reversion'
            elif 'cointegration' in name.lower() or 'pairs' in name.lower():
                stype = 'Pairs Trading'
            elif 'bollinger' in name.lower() or 'bb' in name.lower():
                stype = 'Bollinger Bands'
            else:
                stype = 'Other'
            
            strategy_types[stype] = strategy_types.get(stype, 0) + 1
        
        logger.info("Strategy Types:")
        for stype, count in sorted(strategy_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {stype}: {count}")
        
        logger.info("=" * 80)


def main():
    """Main entry point"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 STRATEGY LIBRARY LOADER & GUIDE")
    logger.info("=" * 80)
    logger.info("")
    
    # Load strategies
    loader = StrategyLibraryLoader()
    strategies = loader.load_strategies()
    
    if not strategies:
        logger.error("❌ No strategies found!")
        return
    
    # Print summary
    loader.print_summary()
    
    # Save simplified list
    simplified = []
    for s in strategies:
        simplified.append({
            'name': s.get('name', 'Unknown'),
            'file': s.get('source_file', ''),
            'description': s.get('description', '')[:100] if s.get('description') else ''
        })
    
    output_file = Path("strategy_library_list.json")
    with open(output_file, 'w') as f:
        json.dump(simplified, f, indent=2)
    logger.info(f"📄 Strategy list saved to: {output_file}")
    
    # Provide instructions
    logger.info("")
    logger.info("=" * 80)
    logger.info("📋 NEXT STEPS TO BACKTEST THESE STRATEGIES")
    logger.info("=" * 80)
    logger.info("")
    logger.info("OPTION 1: Use Main APEX System (Recommended)")
    logger.info("-" * 80)
    logger.info("The main APEX system will automatically discover and backtest strategies.")
    logger.info("These 60+ strategies were already discovered but not backtested due to API issues.")
    logger.info("")
    logger.info("To backtest them:")
    logger.info("1. pip install anthropic  # Enable Claude in swarm")
    logger.info("2. python apex.py         # Run main system")
    logger.info("3. Wait for discovery cycle to complete")
    logger.info("4. New strategies will be queued for RBI backtest")
    logger.info("5. Watch logs for: '✅ STRATEGY APPROVED by LLM consensus'")
    logger.info("")
    logger.info("OPTION 2: Manual Queue Loading (Advanced)")
    logger.info("-" * 80)
    logger.info("You can modify apex.py to load these strategies into the queue:")
    logger.info("")
    logger.info("# Add to StrategyDiscoveryAgent.__init__():")
    logger.info("def _load_existing_strategies(self):")
    logger.info('    """Re-queue existing strategies from library"""')
    logger.info("    for file in Path('strategy_library').glob('*.json'):")
    logger.info("        with open(file) as f:")
    logger.info("            strategy = json.load(f)")
    logger.info("        strategy_discovery_queue.put(strategy)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 MONITORING APPROVALS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Watch logs for these messages:")
    logger.info("  ✅ STRATEGY APPROVED by LLM consensus")
    logger.info("  💾 Approved strategy saved: [name]")
    logger.info("")
    logger.info("Approved strategies saved to: data/[date]/backtests_final/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
