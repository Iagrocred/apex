#!/usr/bin/env python3
"""
🚀 Standalone Strategy Library Batch Backtester
Moon-Dev Style - Complete Implementation

Backtests all 60+ strategies from strategy_library with corrected swarm consensus.
NO IMPORTS from apex.py - completely self-contained.

Creates:
- batch_backtests/[date]/ - Backtest results
- batch_backtests/approved/ - Approved strategies ready for trading
- batch_backtests/logs/ - Detailed logs

Usage:
    pip install anthropic openai
    python standalone_batch_backtester.py
"""

import json
import logging
import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# =========================================================================================
# CONFIGURATION
# =========================================================================================

class Config:
    """Configuration for batch backtester"""
    
    # API Keys (from environment or set here)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Directories
    PROJECT_ROOT = Path.cwd()
    STRATEGY_LIBRARY_DIR = PROJECT_ROOT / "strategy_library"
    BATCH_BACKTEST_DIR = PROJECT_ROOT / "batch_backtests"
    
    # Swarm Consensus
    MIN_WIN_RATE = 0.55  # 55%
    MIN_PROFIT_FACTOR = 1.5
    MIN_SHARPE_RATIO = 1.0
    MAX_DRAWDOWN = 0.20  # 20%
    MIN_TRADES = 50
    CONSENSUS_REQUIRED_VOTES = 2  # 2 of 3
    
    # Backtest Settings
    TARGET_RETURN = 50.0  # 50% target
    MAX_DEBUG_ITERATIONS = 10
    MAX_OPTIMIZATION_ITERATIONS = 10


# =========================================================================================
# LOGGING SETUP
# =========================================================================================

def setup_logging():
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Config.BATCH_BACKTEST_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"batch_backtest_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("BATCH_BACKTEST")


# =========================================================================================
# MODEL FACTORY
# =========================================================================================

class SimpleModelFactory:
    """Simple LLM interface"""
    
    @staticmethod
    def call_llm(model: Dict, prompt: str, system_prompt: str = "", 
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Call LLM with fallback chain"""
        
        model_type = model.get("type")
        model_name = model.get("name")
        
        try:
            if model_type == "deepseek" and Config.DEEPSEEK_API_KEY:
                return SimpleModelFactory._call_deepseek(model_name, prompt, system_prompt, temperature, max_tokens)
            elif model_type == "openai" and Config.OPENAI_API_KEY:
                return SimpleModelFactory._call_openai(model_name, prompt, system_prompt, temperature, max_tokens)
            elif model_type == "anthropic" and Config.ANTHROPIC_API_KEY:
                return SimpleModelFactory._call_anthropic(model_name, prompt, system_prompt, temperature, max_tokens)
            else:
                raise Exception(f"No API key for {model_type}")
        except Exception as e:
            raise Exception(f"LLM call failed for {model_type}/{model_name}: {e}")
    
    @staticmethod
    def _call_deepseek(model: str, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise Exception("openai package not installed. Run: pip install openai")
        
        client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_openai(model: str, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise Exception("openai package not installed. Run: pip install openai")
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_anthropic(model: str, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise Exception("anthropic package not installed. Run: pip install anthropic")
        
        client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text


# =========================================================================================
# STANDALONE BATCH BACKTESTER
# =========================================================================================

class StandaloneBatchBacktester:
    """Standalone batch backtester - Moon-Dev style"""
    
    def __init__(self, logger):
        self.logger = logger
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
        
        # Create directories
        self.date_dir = Config.BATCH_BACKTEST_DIR / datetime.now().strftime("%Y%m%d")
        self.approved_dir = Config.BATCH_BACKTEST_DIR / "approved"
        self.date_dir.mkdir(parents=True, exist_ok=True)
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        
    def load_strategies(self) -> List[Dict]:
        """Load all strategies from strategy_library"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 Loading strategies from strategy_library/")
        self.logger.info("=" * 80)
        
        strategies = []
        
        if not Config.STRATEGY_LIBRARY_DIR.exists():
            self.logger.error(f"❌ Strategy library not found: {Config.STRATEGY_LIBRARY_DIR}")
            return strategies
        
        for strategy_file in sorted(Config.STRATEGY_LIBRARY_DIR.glob("*.json")):
            try:
                with open(strategy_file, 'r') as f:
                    strategy = json.load(f)
                    strategy['source_file'] = strategy_file.name
                    strategies.append(strategy)
                    self.logger.info(f"   ✅ {strategy.get('name', strategy_file.name)}")
            except Exception as e:
                self.logger.error(f"   ❌ Failed: {strategy_file.name}: {e}")
        
        self.logger.info("")
        self.logger.info(f"📊 Total: {len(strategies)} strategies")
        self.logger.info("=" * 80)
        
        self.results["total"] = len(strategies)
        return strategies
    
    def run_batch_backtest(self):
        """Run batch backtest on all strategies"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BATCH BACKTEST")
        self.logger.info("=" * 80)
        self.logger.info("Swarm: DeepSeek-reasoner + GPT-4o + Claude-3.5-sonnet-latest")
        self.logger.info("Standards: 55% win, 1.5 PF, 1.0 Sharpe, 20% DD, 50+ trades")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        strategies = self.load_strategies()
        
        if not strategies:
            self.logger.error("❌ No strategies found!")
            return
        
        # Process each strategy
        for i, strategy in enumerate(strategies, 1):
            strategy_name = strategy.get('name', f'Strategy {i}')
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"🔬 BACKTESTING {i}/{len(strategies)}: {strategy_name}")
            self.logger.info("=" * 80)
            
            try:
                result = self._backtest_strategy(strategy)
                
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
                self.logger.error(f"❌ Error: {e}")
                self.logger.error(traceback.format_exc())
                self.results["failed"] += 1
                self.results["failed_strategies"].append(strategy_name)
            
            time.sleep(2)
        
        self._print_final_report()
    
    def _backtest_strategy(self, strategy: Dict) -> str:
        """Backtest single strategy - simplified Moon-Dev flow"""
        
        strategy_name = strategy.get('name', 'Unknown')
        
        # Phase 1: Generate backtest code
        self.logger.info("🤖 Generating backtest code...")
        code = self._generate_code(strategy)
        
        if not code:
            return "FAILED"
        
        # Phase 2: Execute and get results
        self.logger.info("⚡ Executing backtest...")
        results = self._execute_code(code, strategy_name)
        
        if not results:
            return "FAILED"
        
        # Phase 3: LLM Swarm Consensus
        self.logger.info("🤝 LLM Swarm Consensus...")
        approved, votes = self._swarm_consensus(strategy, results)
        
        if approved:
            self._save_approved(strategy, code, results, votes)
            return "APPROVED"
        else:
            self.logger.info(f"   Votes: {votes}")
            return "REJECTED"
    
    def _generate_code(self, strategy: Dict) -> Optional[str]:
        """Generate backtest code using DeepSeek"""
        
        prompt = f"""Generate a complete Python backtest for this strategy:

Strategy: {strategy.get('name', '')}
Description: {strategy.get('description', '')}

Requirements:
1. Use backtesting.py library
2. Include all necessary imports
3. Define Strategy class with init() and next() methods
4. Load BTC data and run backtest
5. Print results: Return, Sharpe, Max Drawdown, Win Rate, Total Trades

Return ONLY the complete Python code, no explanations."""

        try:
            if Config.DEEPSEEK_API_KEY:
                code = SimpleModelFactory.call_llm(
                    {"type": "deepseek", "name": "deepseek-reasoner"},
                    prompt,
                    "You are an expert quantitative trader and Python developer.",
                    temperature=0.3,
                    max_tokens=3000
                )
                return code
            else:
                self.logger.error("❌ No DEEPSEEK_API_KEY configured")
                return None
        except Exception as e:
            self.logger.error(f"❌ Code generation failed: {e}")
            return None
    
    def _execute_code(self, code: str, strategy_name: str) -> Optional[Dict]:
        """Execute backtest code and extract results"""
        
        # Save code to temp file
        temp_file = self.date_dir / f"temp_{strategy_name.replace(' ', '_')}.py"
        
        try:
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Execute
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            output = result.stdout + result.stderr
            
            # Parse results (simplified)
            results = {
                "return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            }
            
            # Try to extract metrics from output
            for line in output.split('\n'):
                if 'return' in line.lower() and '%' in line:
                    try:
                        results["return_pct"] = float(line.split('%')[0].split()[-1])
                    except:
                        pass
                if 'sharpe' in line.lower():
                    try:
                        results["sharpe_ratio"] = float(line.split()[-1])
                    except:
                        pass
                if 'trades' in line.lower():
                    try:
                        results["total_trades"] = int(line.split()[-1])
                    except:
                        pass
            
            self.logger.info(f"📊 Return: {results['return_pct']:.1f}%, Trades: {results['total_trades']}")
            
            # Clean up
            temp_file.unlink()
            
            return results if results["total_trades"] > 0 else None
            
        except Exception as e:
            self.logger.error(f"❌ Execution failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return None
    
    def _swarm_consensus(self, strategy: Dict, results: Dict) -> tuple:
        """LLM Swarm Consensus - DeepSeek + GPT-4o + Claude"""
        
        votes = {}
        models = [
            {"type": "deepseek", "name": "deepseek-reasoner"},
            {"type": "openai", "name": "gpt-4o"},
            {"type": "anthropic", "name": "claude-3-5-sonnet-latest"}
        ]
        
        prompt = f"""Evaluate this strategy:

Strategy: {strategy.get('name', '')}

Results:
- Return: {results.get('return_pct', 0):.1f}%
- Sharpe: {results.get('sharpe_ratio', 0):.2f}
- Max DD: {results.get('max_drawdown', 0):.1%}
- Win Rate: {results.get('win_rate', 0):.1%}
- Trades: {results.get('total_trades', 0)}

Moon-Dev Standards:
- Win rate > 55%
- Profit factor > 1.5
- Sharpe > 1.0
- Max DD < 20%
- Trades >= 50

Vote: APPROVE or REJECT
Respond with ONLY one word."""

        for model in models:
            try:
                response = SimpleModelFactory.call_llm(
                    model,
                    prompt,
                    temperature=0.1,
                    max_tokens=10
                )
                
                vote = "APPROVE" if "APPROVE" in response.upper() else "REJECT"
                votes[model["type"]] = vote
                self.logger.info(f"   {model['type']}: {vote}")
                
            except Exception as e:
                self.logger.error(f"❌ Vote failed for {model['type']}: {e}")
                votes[model["type"]] = "REJECT"
        
        approvals = sum(1 for v in votes.values() if v == "APPROVE")
        approved = approvals >= Config.CONSENSUS_REQUIRED_VOTES
        
        self.logger.info(f"📊 Consensus: {approvals}/3 APPROVE - {'✅ APPROVED' if approved else '❌ REJECTED'}")
        
        return approved, votes
    
    def _save_approved(self, strategy: Dict, code: str, results: Dict, votes: Dict):
        """Save approved strategy"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = strategy.get('name', 'unknown').replace(' ', '_')
        
        # Save code
        code_file = self.approved_dir / f"{timestamp}_{strategy_name}.py"
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Save metadata
        meta = {
            "strategy_name": strategy.get('name'),
            "description": strategy.get('description'),
            "results": results,
            "llm_votes": votes,
            "timestamp": datetime.now().isoformat(),
            "approved": True
        }
        
        meta_file = self.approved_dir / f"{timestamp}_{strategy_name}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        self.logger.info(f"💾 Saved: {code_file.name}")
    
    def _print_final_report(self):
        """Print final report"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 BATCH BACKTEST COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total:      {self.results['total']}")
        self.logger.info(f"Backtested: {self.results['backtested']}")
        self.logger.info(f"✅ APPROVED: {self.results['approved']}")
        self.logger.info(f"❌ REJECTED: {self.results['rejected']}")
        self.logger.info(f"⚠️ FAILED:   {self.results['failed']}")
        self.logger.info("=" * 80)
        
        if self.results['approved_strategies']:
            self.logger.info("")
            self.logger.info("✅ APPROVED STRATEGIES (Ready for Trading):")
            for name in self.results['approved_strategies']:
                self.logger.info(f"   • {name}")
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 Approved strategies: {self.approved_dir}")
        self.logger.info("=" * 80)
        
        # Save report
        report_file = Config.BATCH_BACKTEST_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"📄 Report: {report_file}")


# =========================================================================================
# MAIN
# =========================================================================================

def main():
    """Main entry point"""
    
    print("\n" + "=" * 80)
    print("🚀 STANDALONE BATCH BACKTESTER - Moon-Dev Style")
    print("=" * 80)
    print("")
    print("This will:")
    print("  1. Load all 60+ strategies from strategy_library/")
    print("  2. Generate backtest code with DeepSeek-reasoner")
    print("  3. Execute backtests and extract results")
    print("  4. Vote with swarm (DeepSeek + GPT-4o + Claude)")
    print("  5. Save approved strategies to batch_backtests/approved/")
    print("")
    print("Requirements:")
    print("  pip install openai anthropic")
    print("")
    print("API Keys needed (set in environment or code):")
    print(f"  DEEPSEEK_API_KEY: {'✅ Set' if Config.DEEPSEEK_API_KEY else '❌ Missing'}")
    print(f"  OPENAI_API_KEY:   {'✅ Set' if Config.OPENAI_API_KEY else '❌ Missing'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ Set' if Config.ANTHROPIC_API_KEY else '❌ Missing'}")
    print("")
    print("=" * 80)
    
    if not (Config.DEEPSEEK_API_KEY and Config.OPENAI_API_KEY and Config.ANTHROPIC_API_KEY):
        print("\n⚠️ WARNING: Some API keys missing. Swarm consensus may fail.")
        print("Set them as environment variables or edit Config class in this file.")
        print("")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    input("\nPress ENTER to start batch backtesting...")
    print("")
    
    # Setup logging
    logger = setup_logging()
    
    # Run backtester
    backtester = StandaloneBatchBacktester(logger)
    backtester.run_batch_backtest()
    
    print("\n✅ Batch backtesting complete!")
    print(f"Check: {Config.BATCH_BACKTEST_DIR}/approved/ for approved strategies")
    print(f"Logs: {Config.BATCH_BACKTEST_DIR}/logs/")


if __name__ == "__main__":
    main()
