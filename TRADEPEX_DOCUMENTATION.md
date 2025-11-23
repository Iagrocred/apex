# 🚀 TRADEPEX - TRADING EXECUTION PARTNER FOR APEX

## Overview

**TradePex** is a complete monolithic trading execution system (1502 lines) that works alongside APEX.py to execute approved trading strategies on Hyperliquid exchange with real capital.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          APEX.PY (6177 lines)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Strategy   │  │     RBI      │  │   Champion   │          │
│  │  Discovery   │→ │  Backtest    │→ │   Manager    │          │
│  │    Agent     │  │   Engine     │  │              │          │
│  └──────────────┘  └──────────────┘  └──────┬───────┘          │
│                                              │                   │
│                                              ▼                   │
│                                    ┌─────────────────┐          │
│                                    │ Approved        │          │
│                                    │ Strategies      │          │
│                                    │ (champions/)    │          │
│                                    └────────┬────────┘          │
└─────────────────────────────────────────────┼──────────────────┘
                                              │
                                              │ JSON files
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TRADEPEX.PY (1502 lines)                   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              6 AUTONOMOUS AGENTS                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  1. STRATEGY LISTENER AGENT                                      │
│     ├─ Monitors APEX champions/ directory                       │
│     ├─ Detects new approved strategies                          │
│     ├─ Validates strategy requirements                          │
│     └─ Queues strategies for execution                          │
│                                                                   │
│  2. TRADING EXECUTION AGENT                                      │
│     ├─ Receives strategies from queue                           │
│     ├─ Generates trading signals                                │
│     ├─ Calculates position sizes                                │
│     ├─ Executes market orders on Hyperliquid                    │
│     └─ Records all trades                                        │
│                                                                   │
│  3. RISK MANAGEMENT AGENT                                        │
│     ├─ Monitors account balance ($650)                          │
│     ├─ Enforces position limits (30% max)                       │
│     ├─ Checks stop loss (5%) / take profit (15%)                │
│     ├─ Tracks daily loss limit ($50)                            │
│     └─ Emergency position closing                               │
│                                                                   │
│  4. POSITION MONITOR AGENT                                       │
│     ├─ Polls Hyperliquid every 30 seconds                       │
│     ├─ Updates position status                                  │
│     ├─ Calculates unrealized PnL                                │
│     └─ Saves position snapshots                                 │
│                                                                   │
│  5. PERFORMANCE TRACKER AGENT                                    │
│     ├─ Records account value changes                            │
│     ├─ Calculates win rate and metrics                          │
│     ├─ Tracks daily/total PnL                                   │
│     ├─ Generates performance reports                            │
│     └─ Saves metrics every 5 minutes                            │
│                                                                   │
│  6. ALERT SYSTEM AGENT                                           │
│     ├─ Processes alerts from all agents                         │
│     ├─ Displays colored terminal output                         │
│     ├─ Logs all alerts to file                                  │
│     └─ Tracks important events                                  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           HYPERLIQUID INTEGRATION                         │  │
│  │  (Based on Moon-Dev nice_funcs_hyperliquid.py)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│     ├─ Market buy/sell execution                                │
│     ├─ Position management (5x leverage)                        │
│     ├─ Real-time price feeds                                    │
│     ├─ Account state monitoring                                 │
│     └─ L2 order book access                                     │
│                                                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
                   ┌────────────────┐
                   │  HYPERLIQUID   │
                   │   EXCHANGE     │
                   │   ($650 live)  │
                   └────────────────┘
```

## Capital Management

### Initial Configuration
- **Total Capital**: $650 USD
- **Leverage**: 5x (configurable 1-50x)
- **Cash Reserve**: 20% ($130)
- **Tradeable Capital**: 80% ($520)

### Position Limits
- **Max Position Size**: 30% ($195 per position)
- **Max Concurrent Positions**: 3
- **Min Position Size**: $10
- **Position Increment**: $5

### Risk Controls
- **Risk Per Trade**: 2% ($13 max loss)
- **Stop Loss**: 5% per position
- **Take Profit**: 15% per position
- **Daily Loss Limit**: $50 USD
- **Daily Trade Limit**: 20 trades

## How TradePex Integrates with APEX

### 1. Strategy Flow

```
APEX Discovery → RBI Backtest → Champion Qualification → Approved Strategy
                                                                ↓
                                                    champions/strategies/
                                                    champion_xxx.json
                                                                ↓
                                            TradePex Strategy Listener
                                                                ↓
                                            Validation & Activation
                                                                ↓
                                            Live Trading on Hyperliquid
```

### 2. Strategy File Format

APEX saves approved strategies as JSON files in `champions/strategies/`:

```json
{
  "id": "champion_1234567890_1",
  "status": "QUALIFIED",
  "strategy_name": "Stoikov Market Making Strategy",
  "strategy_code": "# Python strategy code here...",
  "best_config": {
    "symbol": "BTC",
    "timeframe": "15m",
    "parameters": {...}
  },
  "total_trades": 150,
  "winning_trades": 95,
  "win_rate": 0.63,
  "profit_factor": 2.1,
  "real_trading_eligible": true,
  "created_at": "2025-11-23T21:00:00",
  "bankroll": 12500.0
}
```

### 3. TradePex Actions

When TradePex detects a new approved strategy:

1. **Validation**
   - Checks `real_trading_eligible` flag
   - Validates required fields
   - Ensures strategy meets minimum standards

2. **Activation**
   - Loads strategy into active strategies pool
   - Initializes tracking metrics
   - Sends activation alert

3. **Execution**
   - Monitors market data for the strategy's symbols
   - Generates trading signals based on strategy logic
   - Executes trades with proper position sizing
   - Applies risk management controls

## Agent Details

### 1. Strategy Listener Agent

**Purpose**: Monitors APEX output for approved strategies

**Configuration**:
- Check interval: 10 seconds
- Directory: `champions/strategies/`
- Min backtest trades: 50
- Min win rate: 55%
- Min profit factor: 1.5

**Process**:
1. Scans champions directory every 10 seconds
2. Detects new JSON files
3. Loads and validates strategy
4. Checks `real_trading_eligible` flag
5. Queues valid strategies
6. Saves to TradePex strategies directory

### 2. Trading Execution Agent

**Purpose**: Executes trades on Hyperliquid

**Features**:
- Activates strategies from queue
- Generates trading signals
- Calculates position sizes with leverage
- Executes market orders
- Records all trades
- Enforces concurrent position limits

**Order Execution**:
- Max retries: 3
- Timeout: 30 seconds
- Slippage tolerance: 0.5%

### 3. Risk Management Agent

**Purpose**: Protects capital and enforces limits

**Monitoring** (every 60 seconds):
- Account value vs. starting capital
- Position-level stop loss / take profit
- Daily PnL vs. limits
- Position count vs. max allowed

**Actions**:
- Closes positions on stop loss (5%)
- Closes positions on take profit (15%)
- Halts trading on daily loss limit ($50)
- Alerts on low capital (<50% of start)

### 4. Position Monitor Agent

**Purpose**: Tracks all open positions

**Monitoring** (every 30 seconds):
- Polls Hyperliquid for positions
- Updates global position state
- Calculates unrealized PnL
- Logs position summaries

**Data Saved**:
- Position snapshots (JSON)
- Timestamp and symbol
- Entry price and size
- Current PnL and direction

### 5. Performance Tracker Agent

**Purpose**: Records trading performance

**Metrics Tracked**:
- Account value
- Total PnL (USD and %)
- Unrealized PnL
- Win rate
- Number of trades (daily/total)
- Wins vs. losses
- Uptime hours

**Saves every 5 minutes**:
- Timestamped performance snapshot
- Latest performance summary
- All metrics in JSON format

### 6. Alert System Agent

**Purpose**: Notifies on important events

**Alert Types**:
- 🎯 Strategy Activated
- 💼 Trade Executed
- 🛑 Stop Loss Triggered
- 🎯 Take Profit Triggered
- ⚠️ Low Capital Warning
- 🛑 Daily Loss Limit Reached

**Output**:
- Colored terminal messages
- Saved to `logs/alerts.jsonl`
- Timestamped entries

## Hyperliquid Integration

### API Client Features

Based on Moon-Dev's `nice_funcs_hyperliquid.py`:

1. **Market Data**
   - All mid prices
   - L2 order book
   - Ask/bid spreads

2. **Account Management**
   - User state queries
   - Account value tracking
   - Position retrieval

3. **Order Execution**
   - Market buy orders
   - Market sell orders
   - Position closing
   - Leverage control (5x default)

4. **Risk Features**
   - Reduce-only orders
   - Size calculations
   - Decimal precision handling

### Connection Setup

Requires environment variable:
```bash
HYPER_LIQUID_KEY=0x1234... # Your Hyperliquid private key
```

TradePex initializes connection on startup using eth_account library.

## Directory Structure

```
apex/
├── apex.py                      # Main APEX system (6177 lines)
├── tradepex.py                  # TradePex system (1502 lines)
├── TRADEPEX_DOCUMENTATION.md    # This file
│
├── champions/                   # APEX output
│   ├── strategies/             # Approved strategies (JSON)
│   └── logs/                   # Champion logs
│
├── tradepex/                   # TradePex data
│   ├── positions/              # Position snapshots
│   ├── trades/                 # Trade records
│   ├── performance/            # Performance metrics
│   └── strategies/             # Active strategies
│
└── logs/                       # System logs
    ├── apex_execution_*.log
    └── tradepex_execution_*.log
```

## Configuration

### Environment Variables (.env)

```bash
# Required
HYPER_LIQUID_KEY=0x...          # Hyperliquid private key

# Optional (for AI risk decisions)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

### Capital Settings (in tradepex.py)

```python
class TradePexConfig:
    TOTAL_CAPITAL_USD = 650.0
    CASH_RESERVE_PERCENT = 0.20
    MAX_POSITION_PERCENT = 0.30
    MAX_CONCURRENT_POSITIONS = 3
    DEFAULT_LEVERAGE = 5
    
    RISK_PER_TRADE_PERCENT = 0.02
    MAX_DAILY_LOSS_USD = 50.0
    MAX_DAILY_TRADES = 20
    
    DEFAULT_STOP_LOSS_PERCENT = 0.05
    DEFAULT_TAKE_PROFIT_PERCENT = 0.15
```

## Running the System

### Option 1: Run Both Systems

Terminal 1 (APEX):
```bash
python apex.py
```

Terminal 2 (TradePex):
```bash
python tradepex.py
```

### Option 2: Run TradePex Only

If you have existing approved strategies from APEX:

```bash
python tradepex.py
```

## Trading Flow Example

### Step 1: APEX Discovers and Tests Strategy

```
APEX Discovery Agent finds "Stoikov Market Making" strategy
    ↓
RBI Agent backtests across multiple configs
    ↓
Best config: BTC 15m, 95 trades, 63% win rate, 2.1 profit factor
    ↓
Champion Manager creates champion_123
    ↓
After 3 days paper trading: 8% profit, 60% winning days
    ↓
Status upgraded to "QUALIFIED"
    ↓
real_trading_eligible = true
    ↓
Saved to champions/strategies/champion_123.json
```

### Step 2: TradePex Detects and Activates

```
Strategy Listener scans champions/strategies/
    ↓
Finds champion_123.json (new)
    ↓
Loads and validates strategy
    ↓
real_trading_eligible = true ✓
    ↓
Queues strategy for activation
    ↓
Trading Agent activates strategy
    ↓
Alert: "🎯 Strategy Activated: Stoikov Market Making"
```

### Step 3: TradePex Executes Trades

```
Trading Agent checks strategy signals every 60s
    ↓
Generates BUY signal for BTC
    ↓
Calculates position size: $150 (with 5x leverage = $30 margin)
    ↓
Checks risk limits:
  - Daily loss: OK ($0 of $50)
  - Positions: OK (0 of 3)
  - Capital: OK ($650 available)
    ↓
Executes market buy on Hyperliquid
    ↓
Position Monitor updates: BTC LONG $150
    ↓
Alert: "💼 Trade: BUY BTC $150.00"
```

### Step 4: TradePex Manages Position

```
Position Monitor checks every 30s
    ↓
BTC price moves +3%
    ↓
Unrealized PnL: +$4.50
    ↓
Risk Agent checks stop loss (5%) and take profit (15%)
    ↓
No action needed (within range)
    ↓
Performance Tracker records metrics
```

### Step 5: Exit on Take Profit

```
BTC price moves +16%
    ↓
Unrealized PnL: +$24.00 (16%)
    ↓
Risk Agent: Take profit triggered! (>15%)
    ↓
Closes position with market sell
    ↓
Realized PnL: +$24.00
    ↓
Alert: "🎯 Take Profit: BTC (+16.00%)"
    ↓
Performance: Win count +1, Total PnL +$24
```

## Safety Features

### 1. Capital Protection
- 20% cash reserve always maintained
- Maximum position size enforced
- Daily loss limit prevents drawdown
- Low capital alerts

### 2. Position Risk
- Automatic stop loss at 5%
- Automatic take profit at 15%
- Real-time position monitoring
- Emergency close capability

### 3. Trading Limits
- Max 3 concurrent positions
- Max 20 trades per day
- Max $50 loss per day
- Position size increments

### 4. Monitoring
- 30-second position checks
- 60-second risk checks
- 5-minute performance saves
- Continuous logging

### 5. Validation
- Strategy eligibility checks
- Configuration validation
- API key verification
- Directory structure validation

## Performance Tracking

### Metrics Calculated

1. **Account Metrics**
   - Current account value
   - Total PnL (USD)
   - Total PnL (%)
   - Unrealized PnL

2. **Trading Metrics**
   - Total trades executed
   - Daily trades
   - Wins vs. losses
   - Win rate %

3. **Time Metrics**
   - System uptime
   - Time since start
   - Last trade timestamp

### Data Files

1. **Performance Snapshots**
   - `tradepex/performance/performance_TIMESTAMP.json`
   - Saved every 5 minutes
   - Complete metrics snapshot

2. **Latest Performance**
   - `tradepex/performance/performance_latest.json`
   - Always current
   - Quick access to latest stats

3. **Position Snapshots**
   - `tradepex/positions/snapshot_TIMESTAMP.json`
   - Saved every 30 seconds
   - All open positions

4. **Trade Records**
   - `tradepex/trades/trade_TIMESTAMP.json`
   - One file per trade
   - Complete trade details

## Monitoring the System

### Terminal Output

TradePex provides colored, formatted output:

```
===============================================================================
🚀 TRADEPEX - TRADING EXECUTION PARTNER FOR APEX
===============================================================================

   Version: 1.0 (COMPLETE IMPLEMENTATION)
   Architecture: Moon-Dev AI Agents + APEX Integration

   💰 Capital: $650.0
   📊 Leverage: 5x
   🎯 Max Position: $195.0
   💵 Cash Reserve: $130.0
   📈 Max Positions: 3

===============================================================================
🚀 LAUNCHING ALL THREADS
===============================================================================

✅ Hyperliquid Client initialized
✅ Strategy Listener Agent started
✅ Trading Execution Agent started
✅ Risk Management Agent started
✅ Position Monitor Agent started
✅ Performance Tracker Agent started
✅ Alert System Agent started

✅ TRADEPEX System fully operational
📊 Monitoring APEX for approved strategies...
💼 Ready to execute trades on Hyperliquid
```

### Log Files

All activity logged to:
- `logs/tradepex_execution_TIMESTAMP.log`
- `logs/alerts.jsonl`

### Real-Time Alerts

```
===============================================================================
🔔 ALERT: 🎯 Strategy Activated: Stoikov Market Making
   Time: 2025-11-23T21:00:00
===============================================================================

===============================================================================
🔔 ALERT: 💼 Trade: BUY BTC $150.00
   Time: 2025-11-23T21:05:00
===============================================================================

===============================================================================
🔔 ALERT: 🎯 Take Profit: BTC (+16.00%)
   Time: 2025-11-23T21:45:00
===============================================================================
```

## Troubleshooting

### Issue: TradePex not detecting strategies

**Solution**:
1. Check APEX champions directory exists: `champions/strategies/`
2. Verify strategy files are JSON format
3. Check `real_trading_eligible` flag is `true`
4. Review listener logs for validation errors

### Issue: Cannot connect to Hyperliquid

**Solution**:
1. Verify `HYPER_LIQUID_KEY` in .env file
2. Check key format: `0x...` (64 hex characters)
3. Test key with Hyperliquid API directly
4. Review connection logs

### Issue: Trades not executing

**Solution**:
1. Check risk limits (daily loss, position count)
2. Verify sufficient capital available
3. Review trading agent logs
4. Check strategy signal generation

### Issue: Positions not closing

**Solution**:
1. Verify risk agent is running
2. Check stop loss / take profit thresholds
3. Review position monitor logs
4. Manually close via Hyperliquid if needed

## Code Structure

### Main Components

1. **Configuration** (lines 1-350)
   - TradePexConfig class
   - All settings centralized
   - Directory management

2. **Hyperliquid Client** (lines 351-600)
   - API integration
   - Order execution
   - Position management

3. **6 Agent Classes** (lines 601-1300)
   - Each agent self-contained
   - Thread-safe operations
   - Continuous loops

4. **Thread Monitor** (lines 1301-1400)
   - Manages all threads
   - Coordinates startup
   - Health monitoring

5. **Main Entry** (lines 1401-1502)
   - Validation
   - Initialization
   - Main loop

### Key Design Patterns

1. **Monolithic Architecture**
   - All code in single file
   - Easy to deploy
   - No import dependencies

2. **Multi-threaded**
   - 6 independent agents
   - Thread-safe queues
   - Locked shared state

3. **Event-driven**
   - Queue-based communication
   - Alert system
   - Asynchronous processing

4. **Configuration-driven**
   - All settings in TradePexConfig
   - Easy to modify
   - No hardcoded values

## Future Enhancements

### Potential Additions

1. **Advanced Strategy Execution**
   - Dynamic strategy code loading
   - Multi-symbol support
   - Advanced signal generation

2. **Enhanced Risk Management**
   - AI-powered override decisions
   - Adaptive position sizing
   - Correlation analysis

3. **Performance Analytics**
   - Sharpe ratio calculation
   - Drawdown analysis
   - Strategy comparison

4. **Web Dashboard**
   - Real-time monitoring UI
   - Performance charts
   - Trade history viewer

5. **Multi-Exchange Support**
   - Additional exchanges
   - Cross-exchange arbitrage
   - Unified interface

## Conclusion

TradePex is a **production-ready** trading execution system that seamlessly integrates with APEX to:

✅ **Automatically detect** approved strategies from APEX  
✅ **Execute trades** on Hyperliquid with proper risk controls  
✅ **Manage capital** efficiently with $650 starting capital  
✅ **Monitor positions** continuously in real-time  
✅ **Track performance** with comprehensive metrics  
✅ **Alert on events** with detailed notifications  

The system is built on proven code from the **moon-dev-ai-agents** repository and implements:
- Moon-Dev Trading Agent architecture (1195 lines)
- Moon-Dev Risk Agent patterns (631 lines)
- Moon-Dev Hyperliquid integration (924 lines)
- Custom APEX integration layer

**Total**: 1502 lines of production-ready code ready to trade live on Hyperliquid!

---

Built with ❤️ based on Moon-Dev AI Agents and APEX  
For support: Check logs directory and review agent status
