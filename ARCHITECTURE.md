# APEX + TRADEPEX SYSTEM ARCHITECTURE

## Complete System Overview

This document describes the complete architecture of the APEX + TradePex trading system.

## System Components

### APEX.PY (6177 lines)
**Role**: Strategy Discovery, Backtesting, and Qualification

```
┌─────────────────────────────────────────────────────────────────┐
│                          APEX.PY                                 │
│                     (6177 lines of code)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  THREAD 1: Strategy Discovery Agent                              │
│  ├─ Web search for trading strategies                           │
│  ├─ LLM extraction of strategy details                          │
│  ├─ Quality filtering                                            │
│  └─ Saves to strategy_library/                                  │
│                                                                   │
│  THREAD 2: RBI Backtest Engine                                   │
│  ├─ Picks up strategies from library                            │
│  ├─ Generates Python backtest code (LLM)                        │
│  ├─ Auto-debug with LLM (up to 10 iterations)                   │
│  ├─ Tests multiple configs (symbols, timeframes, periods)       │
│  ├─ Optimization loops                                           │
│  └─ LLM Swarm Consensus (3 LLMs vote)                          │
│                                                                   │
│  THREAD 3: Champion Manager                                      │
│  ├─ Creates "Champions" from approved strategies                │
│  ├─ Paper trading with $10K virtual capital                     │
│  ├─ 3-tier system: CHAMPION → QUALIFIED → ELITE                │
│  ├─ Qualification requirements:                                 │
│  │  • QUALIFIED: 3 days, 50 trades, 60% winning days, 8% profit│
│  │  • ELITE: 14 days, 200 trades, 65% winning days, 25% profit │
│  └─ Sets real_trading_eligible flag                             │
│                                                                   │
│  THREAD 4: Market Data Agents                                    │
│  ├─ Whale Agent (monitors large trades)                         │
│  ├─ Sentiment Agent (Twitter analysis)                          │
│  └─ Funding Agent (funding rate tracking)                       │
│                                                                   │
│  THREAD 5: API Server                                            │
│  └─ FastAPI monitoring dashboard (port 8000)                    │
│                                                                   │
│  OUTPUT:                                                         │
│  └─ champions/strategies/champion_XXX.json                      │
│     {                                                            │
│       "id": "champion_1234567890_1",                            │
│       "status": "QUALIFIED",                                    │
│       "strategy_name": "Stoikov Market Making",                 │
│       "strategy_code": "# Python code...",                      │
│       "best_config": {...},                                     │
│       "total_trades": 150,                                      │
│       "winning_trades": 95,                                     │
│       "win_rate": 0.63,                                         │
│       "profit_factor": 2.1,                                     │
│       "real_trading_eligible": true                             │
│     }                                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### TRADEPEX.PY (1502 lines)
**Role**: Live Trading Execution on Hyperliquid

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRADEPEX.PY                               │
│                     (1502 lines of code)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  AGENT 1: Strategy Listener                                      │
│  ├─ Scans champions/strategies/ every 10 seconds                │
│  ├─ Detects new JSON files from APEX                            │
│  ├─ Validates strategy:                                          │
│  │  • Has real_trading_eligible = true                          │
│  │  • Has strategy_name and best_config                         │
│  │  • Meets minimum requirements                                │
│  └─ Queues for activation                                        │
│                                                                   │
│  AGENT 2: Trading Execution                                      │
│  ├─ Receives strategies from queue                              │
│  ├─ Activates strategy for live trading                         │
│  ├─ Generates trading signals                                   │
│  ├─ Calculates position sizes (with leverage)                   │
│  ├─ Executes market orders on Hyperliquid                       │
│  ├─ Records all trades to tradepex/trades/                      │
│  └─ Respects risk limits                                         │
│                                                                   │
│  AGENT 3: Risk Management                                        │
│  ├─ Monitors account value ($650 capital)                       │
│  ├─ Checks each position every 60 seconds:                      │
│  │  • Stop Loss: Close if -5% loss                              │
│  │  • Take Profit: Close if +15% profit                         │
│  ├─ Enforces daily loss limit ($50)                             │
│  ├─ Enforces position count limit (3 max)                       │
│  ├─ Alerts on low capital (<50% of start)                       │
│  └─ Emergency position closing                                   │
│                                                                   │
│  AGENT 4: Position Monitor                                       │
│  ├─ Polls Hyperliquid API every 30 seconds                      │
│  ├─ Updates position data (size, PnL, direction)                │
│  ├─ Calculates unrealized PnL                                   │
│  ├─ Updates global positions state                              │
│  └─ Saves snapshots to tradepex/positions/                      │
│                                                                   │
│  AGENT 5: Performance Tracker                                    │
│  ├─ Records account value every 5 minutes                       │
│  ├─ Calculates metrics:                                          │
│  │  • Total PnL (USD and %)                                     │
│  │  • Win rate (wins / total)                                   │
│  │  • Daily/total trade counts                                  │
│  ├─ Saves to tradepex/performance/                              │
│  └─ Maintains performance_latest.json                           │
│                                                                   │
│  AGENT 6: Alert System                                           │
│  ├─ Processes alerts from all agents                            │
│  ├─ Displays colored terminal output                            │
│  ├─ Logs to logs/alerts.jsonl                                   │
│  └─ Alert types:                                                 │
│     • Strategy activated                                         │
│     • Trade executed                                             │
│     • Stop loss / take profit triggered                         │
│     • Low capital warning                                        │
│     • Daily loss limit reached                                   │
│                                                                   │
│  HYPERLIQUID CLIENT:                                             │
│  ├─ Market buy/sell orders                                      │
│  ├─ Position management (5x leverage)                           │
│  ├─ Real-time price feeds (L2 book)                             │
│  ├─ Account state queries                                        │
│  └─ Based on moon-dev-ai-agents nice_funcs_hyperliquid.py      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                   ┌────────────────┐
                   │  HYPERLIQUID   │
                   │   EXCHANGE     │
                   │                │
                   │  • Live trading│
                   │  • 5x leverage │
                   │  • $650 capital│
                   └────────────────┘
```

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. STRATEGY DISCOVERY                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        APEX Strategy Discovery Agent
        ├─ Searches web for strategies
        ├─ Extracts strategy details with LLM
        └─ Saves to strategy_library/
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     2. BACKTESTING                               │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        APEX RBI Backtest Engine
        ├─ Generates Python backtest code
        ├─ Auto-debugs code (up to 10 iterations)
        ├─ Tests multiple configurations
        ├─ Optimizes parameters
        └─ LLM Swarm votes (3 LLMs)
                             ↓
                      APPROVED?
                   YES ↓        ↓ NO (rejected)
┌─────────────────────────────────────────────────────────────────┐
│                  3. CHAMPION CREATION                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        APEX Champion Manager
        ├─ Creates "Champion" with $10K virtual
        ├─ Paper trades strategy
        ├─ Tracks daily performance
        └─ Records all trades
                             ↓
              3+ days of paper trading
                             ↓
                      QUALIFIED?
              (60% winning days, 8%+ profit)
                   YES ↓        ↓ NO (continues trading)
                             ↓
         Sets real_trading_eligible = true
                             ↓
         Saves to champions/strategies/champion_XXX.json
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   4. STRATEGY DETECTION                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        TradePex Strategy Listener
        ├─ Scans directory every 10s
        ├─ Finds champion_XXX.json
        ├─ Validates real_trading_eligible
        └─ Queues for activation
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   5. STRATEGY ACTIVATION                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        TradePex Trading Execution Agent
        ├─ Receives from queue
        ├─ Loads strategy code and config
        ├─ Activates for live trading
        └─ Alert: "Strategy Activated"
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    6. SIGNAL GENERATION                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        Strategy Logic Execution
        ├─ Fetches market data for symbol
        ├─ Runs strategy code/logic
        ├─ Generates trading signal:
        │  • BUY (open/maintain position)
        │  • SELL (close position)
        │  • HOLD (no action)
        └─ Calculates signal strength
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   7. RISK VALIDATION                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        TradePex Risk Checks
        ├─ Check daily loss limit ($50)
        ├─ Check daily trade limit (20)
        ├─ Check position count (3 max)
        ├─ Check available capital
        └─ Calculate position size
                             ↓
                    ALL CHECKS PASS?
                   YES ↓        ↓ NO (skip trade)
┌─────────────────────────────────────────────────────────────────┐
│                    8. TRADE EXECUTION                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        Hyperliquid Client
        ├─ Get current price (ask/bid)
        ├─ Calculate contracts (size / price)
        ├─ Place market order
        ├─ Record trade to tradepex/trades/
        └─ Alert: "Trade Executed"
                             ↓
                      HYPERLIQUID EXCHANGE
                      (Order fills instantly)
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   9. POSITION MONITORING                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        TradePex Position Monitor (every 30s)
        ├─ Query Hyperliquid for positions
        ├─ Update position data
        ├─ Calculate unrealized PnL
        └─ Save snapshot
                             ↓
        TradePex Risk Management (every 60s)
        ├─ Check each position:
        │  • If PnL <= -5%: STOP LOSS → Close
        │  • If PnL >= +15%: TAKE PROFIT → Close
        ├─ Check account value
        └─ Check daily loss
                             ↓
                    CLOSE POSITION?
              YES ↓                 ↓ NO (continue monitoring)
                             ↓
        Execute Close Order
        ├─ Market sell (if long)
        ├─ Market buy (if short)
        ├─ Record realized PnL
        └─ Alert: "Stop Loss" or "Take Profit"
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  10. PERFORMANCE TRACKING                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
        TradePex Performance Tracker (every 5min)
        ├─ Query account value
        ├─ Calculate total PnL
        ├─ Calculate win rate
        ├─ Count trades (daily/total)
        ├─ Save to tradepex/performance/
        └─ Update performance_latest.json
                             ↓
                      CONTINUOUS LOOP
                      (back to step 6)
```

## Capital Management Details

### $650 Capital Allocation

```
Total Capital: $650
├─ Cash Reserve (20%): $130
│  └─ Never traded, safety buffer
│
└─ Tradeable Capital (80%): $520
   ├─ Max per position (30%): $195
   ├─ With 5x leverage:
   │  └─ $195 position = $39 margin required
   │
   └─ Max concurrent positions: 3
      ├─ Position 1: up to $195 ($39 margin)
      ├─ Position 2: up to $195 ($39 margin)
      └─ Position 3: up to $195 ($39 margin)
      
Total if fully invested:
  $585 notional (3 × $195)
  $117 margin used (3 × $39)
  $533 available margin ($650 - $117)
```

### Risk Controls per Position

```
Position Size: $195 (30% of $650)

With 5x Leverage:
├─ Notional value: $195
├─ Margin required: $39 ($195 / 5)
└─ Liquidation: ~80% loss (due to 5x leverage)

Stop Loss (5%):
├─ Loss amount: $9.75 ($195 × 0.05)
├─ Returns margin: $39 - $9.75 = $29.25
└─ Account after: $650 - $9.75 = $640.25

Take Profit (15%):
├─ Profit amount: $29.25 ($195 × 0.15)
├─ Returns margin: $39 + $29.25 = $68.25
└─ Account after: $650 + $29.25 = $679.25
```

## File Structure

```
apex/
│
├── apex.py (6177 lines)
│   └── Discovers, backtests, qualifies strategies
│
├── tradepex.py (1502 lines)
│   └── Executes trades on Hyperliquid
│
├── TRADEPEX_DOCUMENTATION.md
│   └── Complete technical documentation
│
├── README_TRADEPEX.md
│   └── Quick start guide
│
├── ARCHITECTURE.md (this file)
│   └── Complete system architecture
│
├── requirements_tradepex.txt
│   └── Python dependencies
│
├── champions/
│   ├── strategies/
│   │   ├── champion_1234567890_1.json ← APEX writes here
│   │   └── champion_1234567890_2.json    TradePex reads from here
│   └── logs/
│
├── tradepex/
│   ├── positions/
│   │   └── snapshot_TIMESTAMP.json
│   ├── trades/
│   │   └── trade_TIMESTAMP.json
│   ├── performance/
│   │   ├── performance_TIMESTAMP.json
│   │   └── performance_latest.json
│   └── strategies/
│       └── champion_XXX.json (active strategies)
│
└── logs/
    ├── apex_execution_TIMESTAMP.log
    ├── tradepex_execution_TIMESTAMP.log
    └── alerts.jsonl
```

## Communication Between Systems

### Shared Directory: `champions/strategies/`

APEX writes to this directory when a strategy qualifies:

```json
{
  "id": "champion_1234567890_1",
  "status": "QUALIFIED",
  "strategy_name": "Strategy Name",
  "strategy_code": "# Python code",
  "best_config": {
    "symbol": "BTC",
    "timeframe": "15m",
    "parameters": {...}
  },
  "real_trading_eligible": true,  ← TradePex checks this flag
  "total_trades": 150,
  "winning_trades": 95,
  "win_rate": 0.63,
  "profit_factor": 2.1,
  "created_at": "2025-11-23T21:00:00"
}
```

TradePex reads from this directory every 10 seconds:
1. Scans for new JSON files
2. Checks `real_trading_eligible` flag
3. Validates strategy meets requirements
4. Queues for activation if valid

### No Direct Communication

- APEX and TradePex don't communicate directly
- They operate independently
- Shared filesystem is the integration point
- TradePex can run without APEX (if strategies exist)
- APEX can run without TradePex (paper trading only)

## Monitoring Both Systems

### Terminal Setup

**Terminal 1 - APEX**:
```bash
python apex.py
# Outputs:
# - Strategy discoveries
# - Backtest results
# - Champion qualifications
# - Paper trading activity
```

**Terminal 2 - TradePex**:
```bash
python tradepex.py
# Outputs:
# - Strategy detections
# - Trade executions
# - Position updates
# - Performance metrics
```

**Terminal 3 - Logs**:
```bash
tail -f logs/tradepex_execution_*.log
# Real-time log monitoring
```

### Key Metrics to Watch

**APEX**:
- Strategies discovered per hour
- Backtest success rate
- Champions created
- Paper trading PnL

**TradePex**:
- Active strategies
- Open positions (0-3)
- Realized PnL
- Win rate
- Daily loss (vs $50 limit)

## Safety and Risk Management

### Multi-Layer Protection

1. **APEX Layer**:
   - Only promotes strategies with proven backtest results
   - Requires 3+ days paper trading
   - 60%+ winning days required
   - 8%+ profit required

2. **TradePex Layer**:
   - Position size limits (30% max)
   - Stop loss on every position (5%)
   - Take profit on every position (15%)
   - Daily loss limit ($50)
   - Position count limit (3)
   - Cash reserve maintained (20%)

3. **Hyperliquid Layer**:
   - Exchange-level liquidation protection
   - Real-time margin monitoring
   - Market order execution

### What Can Go Wrong

1. **Strategy Performs Poorly Live**:
   - Stop loss will limit loss to 5% per position
   - Daily loss limit will halt trading at $50
   - Manual intervention possible

2. **Market Volatility**:
   - Leverage amplifies both gains and losses
   - Stop loss may execute at worse price in fast market
   - Position monitor checks every 30s

3. **Capital Depletion**:
   - TradePex alerts if capital drops <50%
   - Will halt trading if limits hit
   - Manual review required

4. **Technical Failures**:
   - All agents independent - others continue if one fails
   - Comprehensive logging
   - Thread monitoring
   - Manual position closing via Hyperliquid interface

## Conclusion

This system combines:

✅ **APEX (6177 lines)**: Automated strategy discovery and qualification  
✅ **TradePex (1502 lines)**: Live trading execution with strict risk controls  
✅ **Moon-Dev Integration**: Based on proven trading agent architecture  
✅ **Capital Safety**: Multiple layers of protection for $650 capital  
✅ **Monitoring**: Comprehensive logging and performance tracking  

**Total System**: 7679 lines of production-ready code

---

**Next Steps**:
1. Run APEX to discover and qualify strategies
2. Once strategies qualify, run TradePex
3. Monitor both systems carefully
4. Review performance daily
5. Adjust risk parameters based on results

**Remember**: This is real money trading. Start conservative, monitor closely, and never trade more than you can afford to lose.
