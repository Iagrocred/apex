# TradePex Quick Start Guide

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Environment Variables**:
   ```bash
   HYPER_LIQUID_KEY=0x...  # Required - Your Hyperliquid private key
   OPENAI_API_KEY=sk-...    # Optional - For AI risk decisions
   ANTHROPIC_API_KEY=sk-... # Optional
   ```

3. **Python Packages**:
   ```bash
   pip install numpy pandas python-dotenv termcolor eth-account requests
   ```

### Installation

1. **Clone/Copy Files**:
   ```bash
   # You should have:
   - apex.py (6177 lines)
   - tradepex.py (1502 lines)
   - TRADEPEX_DOCUMENTATION.md (full docs)
   - README_TRADEPEX.md (this file)
   ```

2. **Set Up Environment**:
   ```bash
   # Create .env file
   echo "HYPER_LIQUID_KEY=0x..." > .env
   ```

3. **Verify Structure**:
   ```bash
   # Directories will be auto-created:
   champions/strategies/  # APEX writes approved strategies here
   tradepex/             # TradePex data
   logs/                 # System logs
   ```

### Running TradePex

#### Option 1: With APEX (Full System)

**Terminal 1** - Run APEX for strategy discovery/testing:
```bash
python apex.py
```

**Terminal 2** - Run TradePex for live trading:
```bash
python tradepex.py
```

#### Option 2: TradePex Only (With Existing Strategies)

If you already have approved strategies from APEX:

```bash
python tradepex.py
```

### What Happens on Startup

```
1. ✅ Validates configuration and API keys
2. ✅ Creates all required directories
3. ✅ Initializes Hyperliquid connection
4. ✅ Starts 6 autonomous agents:
   - Strategy Listener
   - Trading Execution
   - Risk Management
   - Position Monitor
   - Performance Tracker
   - Alert System
5. ✅ Begins monitoring APEX champions directory
6. ✅ Ready to execute trades!
```

## 📊 How It Works

### The Flow

```
APEX discovers strategy
        ↓
RBI backtests strategy
        ↓
Champion qualifies (3+ days, 8%+ profit)
        ↓
Saved to champions/strategies/champion_XXX.json
        ↓
TradePex detects new file (every 10s)
        ↓
Validates strategy eligibility
        ↓
Activates for live trading
        ↓
Executes trades on Hyperliquid
        ↓
Manages risk and tracks performance
```

### Capital Management ($650)

| Component | Amount | Percentage |
|-----------|--------|------------|
| Total Capital | $650 | 100% |
| Cash Reserve | $130 | 20% |
| Tradeable Capital | $520 | 80% |
| Max Position | $195 | 30% |
| Max Concurrent Positions | 3 | - |

**With 5x Leverage**:
- $195 position = $39 margin required
- $520 tradeable = can support 3x $195 positions comfortably

### Risk Controls

✅ **Stop Loss**: Automatic close at -5%  
✅ **Take Profit**: Automatic close at +15%  
✅ **Daily Loss Limit**: $50 maximum  
✅ **Position Limit**: 3 concurrent max  
✅ **Trade Limit**: 20 per day  

## 🎯 Example Session

### Startup
```bash
$ python tradepex.py

===============================================================================
🚀 TRADEPEX - TRADING EXECUTION PARTNER FOR APEX
===============================================================================
   💰 Capital: $650.0
   📊 Leverage: 5x
   🎯 Max Position: $195.0
   📈 Max Positions: 3
===============================================================================
🚀 LAUNCHING ALL THREADS
===============================================================================
✅ Hyperliquid Client initialized
✅ All threads started successfully
✅ TRADEPEX System fully operational
📊 Monitoring APEX for approved strategies...
💼 Ready to execute trades on Hyperliquid
```

### Strategy Detection
```
===============================================================================
🔔 ALERT: 🎯 Strategy Activated: Stoikov Market Making Strategy
   Time: 2025-11-23T21:00:00
===============================================================================
```

### Trade Execution
```
===============================================================================
🎯 EXECUTING TRADE
   Symbol: BTC
   Direction: BUY
   Size: $150.00
   Strategy: champion_1234567890_1
===============================================================================
✅ Trade executed successfully

===============================================================================
🔔 ALERT: 💼 Trade: BUY BTC $150.00
   Time: 2025-11-23T21:05:00
===============================================================================
```

### Position Monitoring
```
===============================================================================
📊 POSITION UPDATE
   Open Positions: 1
   Total Unrealized PnL: $4.50
   BTC: LONG $150.00 | PnL: $4.50
===============================================================================
```

### Performance Update
```
===============================================================================
📈 PERFORMANCE UPDATE
   Account Value: $654.50
   Total PnL: $4.50 (+0.69%)
   Win Rate: 100.0% (1W / 0L)
   Total Trades: 1
===============================================================================
```

### Take Profit
```
===============================================================================
🔔 ALERT: 🎯 Take Profit: BTC (+16.00%)
   Time: 2025-11-23T21:45:00
===============================================================================
```

## 📁 Generated Files

### Position Snapshots
`tradepex/positions/snapshot_TIMESTAMP.json`
```json
{
  "timestamp": "2025-11-23T21:30:00",
  "positions": [
    {
      "coin": "BTC",
      "size": 150.0,
      "is_long": true,
      "entry_price": 45000.0,
      "unrealized_pnl": 4.50
    }
  ],
  "count": 1,
  "total_unrealized_pnl": 4.50
}
```

### Trade Records
`tradepex/trades/trade_TIMESTAMP.json`
```json
{
  "timestamp": "2025-11-23T21:05:00",
  "symbol": "BTC",
  "direction": "BUY",
  "size_usd": 150.0,
  "strategy_id": "champion_1234567890_1",
  "price": 45000.0
}
```

### Performance Metrics
`tradepex/performance/performance_latest.json`
```json
{
  "timestamp": "2025-11-23T21:30:00",
  "account_value": 654.50,
  "total_pnl_usd": 4.50,
  "total_pnl_percent": 0.69,
  "win_rate": 100.0,
  "total_trades": 1,
  "wins": 1,
  "losses": 0
}
```

## ⚙️ Configuration

Edit `tradepex.py` to adjust settings:

```python
class TradePexConfig:
    # Capital
    TOTAL_CAPITAL_USD = 650.0          # Your total capital
    CASH_RESERVE_PERCENT = 0.20        # 20% reserve
    MAX_POSITION_PERCENT = 0.30        # 30% per position
    MAX_CONCURRENT_POSITIONS = 3       # Max positions
    
    # Leverage
    DEFAULT_LEVERAGE = 5               # 1-50x
    
    # Risk
    RISK_PER_TRADE_PERCENT = 0.02     # 2% risk
    MAX_DAILY_LOSS_USD = 50.0         # $50 daily limit
    DEFAULT_STOP_LOSS_PERCENT = 0.05  # 5% stop loss
    DEFAULT_TAKE_PROFIT_PERCENT = 0.15 # 15% take profit
    
    # Monitoring
    STRATEGY_CHECK_INTERVAL_SECONDS = 10  # Check for new strategies
    POSITION_CHECK_INTERVAL_SECONDS = 30  # Check positions
    RISK_CHECK_INTERVAL_SECONDS = 60      # Risk checks
```

## 🔍 Monitoring

### Check System Status

**Logs**:
```bash
tail -f logs/tradepex_execution_*.log
```

**Alerts**:
```bash
tail -f logs/alerts.jsonl
```

**Latest Performance**:
```bash
cat tradepex/performance/performance_latest.json | python -m json.tool
```

**Open Positions**:
```bash
ls -lt tradepex/positions/ | head -5
```

### What to Watch

✅ **Green** = Good (Trades, profits, successful operations)  
⚠️ **Yellow** = Warning (Alerts, limits approaching)  
❌ **Red** = Error (Failed trades, risk breaches)  

## 🛑 Stopping TradePex

Press `Ctrl+C` in the terminal:

```
🛑 Shutting down TradePex...
Goodbye!
```

**What happens**:
- All threads gracefully stop
- Final performance metrics saved
- Open positions remain on Hyperliquid (not auto-closed)
- All logs flushed to disk

**To close positions**: Either restart TradePex (risk agent will manage them) or close manually via Hyperliquid

## 🐛 Troubleshooting

### No strategies detected

**Check**:
```bash
ls -la champions/strategies/
```

**Expected**: JSON files with `real_trading_eligible: true`

**Fix**: Run APEX first to generate strategies, or manually place strategy JSON files

### Connection errors

**Check**:
```bash
echo $HYPER_LIQUID_KEY
```

**Expected**: `0x...` format (66 characters)

**Fix**: Verify key in .env file and Hyperliquid account access

### Trades not executing

**Check logs**:
```bash
grep -i "risk limit\|daily loss\|position count" logs/tradepex_execution_*.log
```

**Common reasons**:
- Daily loss limit reached ($50)
- Max positions reached (3)
- Insufficient capital
- Strategy signals not generating

### Positions not closing

**Manual check**:
```bash
# Check if risk agent is running
ps aux | grep tradepex
```

**Fix**: Restart TradePex or manually close positions on Hyperliquid interface

## 📞 Support

### Check These First

1. **Logs**: `logs/tradepex_execution_*.log`
2. **Alerts**: `logs/alerts.jsonl`
3. **Documentation**: `TRADEPEX_DOCUMENTATION.md`
4. **Performance**: `tradepex/performance/performance_latest.json`

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Install missing packages: `pip install ...` |
| Key errors | Check .env file and HYPER_LIQUID_KEY format |
| No strategies | Run APEX or manually create strategy JSON |
| Positions stuck | Restart TradePex or close manually |
| High losses | Check risk settings and stop loss values |

## 🎓 Understanding the System

### Agent Roles

1. **Strategy Listener** 👂
   - Watches for new strategies from APEX
   - Runs every 10 seconds
   - Validates before activation

2. **Trading Execution** 💼
   - Executes trades on Hyperliquid
   - Manages active strategies
   - Records all trades

3. **Risk Management** 🛡️
   - Enforces all limits
   - Closes positions on stop/take profit
   - Monitors daily loss

4. **Position Monitor** 📊
   - Updates position data
   - Calculates PnL
   - Saves snapshots

5. **Performance Tracker** 📈
   - Records metrics
   - Calculates win rate
   - Generates reports

6. **Alert System** 🔔
   - Processes alerts
   - Colored output
   - Logs events

### Data Flow

```
champions/strategies/ → Strategy Listener → Trading Execution
                                                    ↓
                                          Hyperliquid Exchange
                                                    ↓
                                          Position Monitor
                                                    ↓
                                       Risk Management ← Performance Tracker
                                                    ↓
                                             Alert System
```

## 💡 Tips

### Best Practices

✅ **Start small**: Test with minimum positions first  
✅ **Monitor closely**: Watch the first few trades carefully  
✅ **Check logs**: Review logs daily for any issues  
✅ **Adjust limits**: Modify risk settings based on results  
✅ **Backup data**: Save performance files regularly  

### Risk Management

⚠️ **Never trade more than you can afford to lose**  
⚠️ **Start with lower leverage (2-3x) until comfortable**  
⚠️ **Keep cash reserve at 20% minimum**  
⚠️ **Review performance daily**  
⚠️ **Adjust limits if hitting them frequently**  

### Performance Optimization

📈 **Let winners run**: Take profit at 15% is aggressive, consider raising  
📉 **Cut losses quickly**: 5% stop loss is good starting point  
🎯 **Quality over quantity**: Better 3 good trades than 20 bad ones  
⏰ **Be patient**: Wait for high-quality strategy signals  

## 🚀 Next Steps

1. **Run first time**:
   ```bash
   python tradepex.py
   ```

2. **Monitor for 24 hours**:
   - Check alerts
   - Review performance
   - Verify risk controls

3. **Adjust settings**:
   - Based on performance
   - Your risk tolerance
   - Market conditions

4. **Scale up**:
   - Once comfortable
   - Proven track record
   - Consistent results

---

**Ready to start?**

```bash
python tradepex.py
```

**Good luck and trade safely! 🌙**

---

Built based on:
- APEX.py (6177 lines)
- Moon-Dev AI Agents (moon-dev-ai-agents repo)
- Hyperliquid integration

For full documentation: See `TRADEPEX_DOCUMENTATION.md`
