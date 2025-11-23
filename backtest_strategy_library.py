#!/usr/bin/env python3
"""
Strategy Library Batch Backtester
Backtests all strategies from strategy_library with corrected swarm consensus

This script:
1. Loads all strategies from strategy_library/
2. Feeds them to RBI for backtesting
3. Uses corrected swarm (DeepSeek + GPT-4o + Claude) for approval
4. Reports which strategies pass Moon-Dev's standards
5. Approved strategies saved to data/[date]/backtests_final/
"""

import json
import logging
import queue
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from apex.py
from apex import (
    RBIBacktestEngine,
    validated_strategy_queue,
    setup_enhanced_logging
)

# Setup logging
logger = setup_enhanced_logging()

class StrategyLibraryBatchBacktester:
    """Batch backtest all strategies from strategy_library"""
    
    def __init__(self):
        self.logger = logging.getLogger("STRATEGY_BATCH")
        self.strategy_library_dir = Path("strategy_library")
        self.results = {
            "total": 0,
            "backtested": 0,
            "approved": 0,
            "rejected": 0,
            "failed": 0,
            "approved_strategies": [],
            "rejected_strategies": [],
            "failed_strategies": []
        }
        
    def load_strategies(self):
        """Load all JSON strategies from strategy_library"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 Loading strategies from strategy_library/")
        self.logger.info("=" * 80)
        
        strategies = []
        
        if not self.strategy_library_dir.exists():
            self.logger.error(f"❌ Strategy library not found: {self.strategy_library_dir}")
            return strategies
            
        for strategy_file in sorted(self.strategy_library_dir.glob("*.json")):
            try:
                with open(strategy_file, 'r') as f:
                    strategy = json.load(f)
                    
                # Add source information
                strategy['source_file'] = strategy_file.name
                strategy['batch_backtest'] = True
                
                strategies.append(strategy)
                self.logger.info(f"   ✅ Loaded: {strategy.get('name', strategy_file.name)}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to load {strategy_file.name}: {e}")
                
        self.results["total"] = len(strategies)
        self.logger.info("")
        self.logger.info(f"📊 Total strategies loaded: {len(strategies)}")
        self.logger.info("=" * 80)
        
        return strategies
        
    def run_batch_backtest(self):
        """Run batch backtest on all strategies"""
        self.logger.info("")
        self.logger.info("🚀 STARTING BATCH BACKTEST")
        self.logger.info("=" * 80)
        self.logger.info("This will backtest ALL strategies from strategy_library/")
        self.logger.info("Using corrected swarm: DeepSeek + GPT-4o + Claude")
        self.logger.info("Moon-Dev Standards: 55% win, 1.5 PF, 1.0 Sharpe, 20% DD, 50+ trades")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # Load strategies
        strategies = self.load_strategies()
        
        if not strategies:
            self.logger.error("❌ No strategies found to backtest!")
            return
            
        # Create RBI engine
        rbi = RBIBacktestEngine()
        
        # Process each strategy
        for i, strategy in enumerate(strategies, 1):
            strategy_name = strategy.get('name', f'Strategy {i}')
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"🔬 BACKTESTING {i}/{len(strategies)}: {strategy_name}")
            self.logger.info("=" * 80)
            
            try:
                # Run RBI backtest process
                result = self._backtest_single_strategy(rbi, strategy)
                
                if result == "APPROVED":
                    self.results["approved"] += 1
                    self.results["approved_strategies"].append(strategy_name)
                    self.logger.info(f"✅ APPROVED: {strategy_name}")
                elif result == "REJECTED":
                    self.results["rejected"] += 1
                    self.results["rejected_strategies"].append(strategy_name)
                    self.logger.info(f"❌ REJECTED: {strategy_name}")
                else:
                    self.results["failed"] += 1
                    self.results["failed_strategies"].append(strategy_name)
                    self.logger.error(f"⚠️ FAILED: {strategy_name}")
                    
                self.results["backtested"] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error backtesting {strategy_name}: {e}")
                self.results["failed"] += 1
                self.results["failed_strategies"].append(strategy_name)
                
            # Small delay between strategies
            time.sleep(2)
            
        # Print final report
        self._print_final_report()
        
    def _backtest_single_strategy(self, rbi: RBIBacktestEngine, strategy: dict):
        """Backtest a single strategy"""
        
        try:
            # Phase 1: Research
            research = rbi._research_strategy(strategy)
            
            # Phase 2: Generate backtest code
            self.logger.info("🤖 Generating backtest code...")
            code = rbi._generate_backtest_code(strategy, research)
            
            # Phase 3: Auto-debug loop
            self.logger.info("🔧 Starting auto-debug loop...")
            executable_code = rbi._auto_debug_loop(code, strategy)
            
            if not executable_code:
                self.logger.warning("❌ Auto-debug failed")
                return "FAILED"
                
            # Phase 4: Execute backtest
            self.logger.info("⚡ Executing backtest...")
            results = rbi._execute_backtest(executable_code)
            
            if not results:
                self.logger.warning("❌ Backtest execution failed")
                return "FAILED"
                
            # Phase 5: Optimization (if needed)
            if results.get("return_pct", 0) < rbi.TARGET_RETURN:
                self.logger.info("🔄 Starting optimization...")
                optimized_code, optimized_results = rbi._optimization_loop(
                    executable_code, strategy, results
                )
                if optimized_results:
                    executable_code = optimized_code
                    results = optimized_results
                    
            # Phase 6: Multi-configuration testing
            self.logger.info("📊 Multi-configuration testing...")
            config_results = rbi._multi_config_testing(executable_code, strategy)
            
            # Phase 7: LLM Swarm Consensus
            self.logger.info("🤝 LLM Swarm Consensus Voting...")
            approved, votes, best_config = rbi._llm_swarm_consensus(
                config_results, strategy, results
            )
            
            if approved:
                self.logger.info(f"✅ STRATEGY APPROVED by LLM consensus")
                self.logger.info(f"   Votes: {votes}")
                
                # Save approved strategy
                validated_strategy = {
                    "strategy_name": strategy.get("name", "Unknown"),
                    "strategy_data": strategy,
                    "code": executable_code,
                    "best_config": best_config,
                    "results": results,
                    "llm_votes": votes,
                    "timestamp": datetime.now().isoformat()
                }
                
                rbi._save_approved_strategy(validated_strategy)
                validated_strategy_queue.put(validated_strategy)
                
                return "APPROVED"
            else:
                self.logger.info(f"❌ STRATEGY REJECTED by LLM consensus")
                self.logger.info(f"   Votes: {votes}")
                return "REJECTED"
                
        except Exception as e:
            self.logger.error(f"❌ Backtest error: {e}")
            return "FAILED"
            
    def _print_final_report(self):
        """Print final batch backtest report"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 BATCH BACKTEST COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Strategies:     {self.results['total']}")
        self.logger.info(f"Backtested:           {self.results['backtested']}")
        self.logger.info(f"✅ APPROVED:          {self.results['approved']}")
        self.logger.info(f"❌ REJECTED:          {self.results['rejected']}")
        self.logger.info(f"⚠️ FAILED:            {self.results['failed']}")
        self.logger.info("=" * 80)
        
        if self.results['approved_strategies']:
            self.logger.info("")
            self.logger.info("✅ APPROVED STRATEGIES:")
            for name in self.results['approved_strategies']:
                self.logger.info(f"   • {name}")
                
        if self.results['rejected_strategies']:
            self.logger.info("")
            self.logger.info("❌ REJECTED STRATEGIES:")
            for name in self.results['rejected_strategies']:
                self.logger.info(f"   • {name}")
                
        if self.results['failed_strategies']:
            self.logger.info("")
            self.logger.info("⚠️ FAILED STRATEGIES:")
            for name in self.results['failed_strategies']:
                self.logger.info(f"   • {name}")
                
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Approved strategies saved to: data/[date]/backtests_final/")
        self.logger.info("=" * 80)
        
        # Save report to file
        report_file = Path("strategy_batch_backtest_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"📄 Report saved to: {report_file}")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("🚀 STRATEGY LIBRARY BATCH BACKTESTER")
    print("=" * 80)
    print("This will backtest ALL 60+ strategies from strategy_library/")
    print("Using corrected swarm consensus (DeepSeek + GPT-4o + Claude)")
    print("=" * 80)
    print("\n⚠️  IMPORTANT:")
    print("1. Make sure you've installed: pip install anthropic")
    print("2. Make sure your API keys are configured in Config")
    print("3. This will take several hours (10-15 min per strategy)")
    print("=" * 80)
    
    input("\nPress ENTER to start batch backtesting...")
    
    # Create and run backtester
    backtester = StrategyLibraryBatchBacktester()
    backtester.run_batch_backtest()
    
    print("\n✅ Batch backtesting complete!")
    print("Check logs/apex_execution_*.log for detailed results")
    print("Approved strategies in: data/[date]/backtests_final/")


if __name__ == "__main__":
    main()
