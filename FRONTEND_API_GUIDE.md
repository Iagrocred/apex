# 🖥️ TRADEPEX ADAPTIVE - FRONTEND API GUIDE

## Base URL
```
Production: http://157.180.54.22:8000
Development: http://localhost:8000
```

---

## 📊 ENDPOINTS OVERVIEW

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status (cycle, capital, mode) |
| GET | `/api/strategies` | All strategies with performance |
| GET | `/api/positions` | Current open positions |
| GET | `/api/trades` | Trade history (paginated) |
| GET | `/api/portfolio` | Portfolio P&L summary |
| GET | `/api/ready-for-live` | Strategies at 71%+ ready for live |
| GET | `/api/llm-reasoning` | LLM brain activity (JSON reasoning) |
| GET | `/api/llm-reasoning/{strategy_id}` | LLM reasoning for specific strategy |
| GET | `/api/versions/{strategy_id}` | All versions of a strategy |
| POST | `/api/go-live/{strategy_id}` | Promote strategy to live trading |
| GET | `/api/config` | Current system configuration |

---

## 🔌 ENDPOINT DETAILS

### 1. GET `/api/status`
**System status and health check**

**Response:**
```json
{
  "status": "RUNNING",
  "mode": "PAPER",
  "cycle": 847,
  "runtime": "04:23:15",
  "capital": {
    "type_a": {
      "initial": 17000.0,
      "current": 17845.50,
      "pnl": 845.50,
      "pnl_percent": 4.97
    },
    "type_b": {
      "initial": 17000.0,
      "current": 18234.25,
      "pnl": 1234.25,
      "pnl_percent": 7.26
    },
    "total": {
      "initial": 34000.0,
      "current": 36079.75,
      "pnl": 2079.75,
      "pnl_percent": 6.12
    }
  },
  "positions": {
    "open": 12,
    "max": 40
  },
  "strategies": {
    "total": 10,
    "active": 8,
    "paused": 2,
    "ready_for_live": 3
  },
  "llm": {
    "provider": "deepseek",
    "last_reasoning": "2024-12-02T01:15:00Z",
    "total_recodes": 156
  }
}
```

---

### 2. GET `/api/strategies`
**All strategies with full performance data**

**Response:**
```json
{
  "strategies": [
    {
      "id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
      "name": "Market Maker Inventory Rebalancing",
      "type": "MARKET_MAKING",
      "status": "ACTIVE",
      "version": "v23",
      "mode": "TYPE_B",
      "leverage": 8,
      "performance": {
        "trades": 67,
        "wins": 49,
        "losses": 18,
        "win_rate": 0.731,
        "profit_factor": 2.14,
        "net_profit": 847.50,
        "avg_win": 28.45,
        "avg_loss": -18.23,
        "max_drawdown": -156.80,
        "consecutive_wins": 7,
        "consecutive_losses": 2
      },
      "ready_for_live": true,
      "ready_for_live_criteria": {
        "trades": { "required": 50, "current": 67, "passed": true },
        "win_rate": { "required": 0.71, "current": 0.731, "passed": true },
        "profit_factor": { "required": 1.8, "current": 2.14, "passed": true },
        "net_profit": { "required": 500, "current": 847.50, "passed": true }
      },
      "llm_iterations": 23,
      "last_recode": "2024-12-02T00:45:00Z",
      "created_at": "2024-11-24T06:18:12Z"
    },
    {
      "id": "20251124_034046_AI_Neural_Network_Predictive_Model",
      "name": "AI Neural Network Predictive",
      "type": "ML_PREDICTION",
      "status": "OPTIMIZING",
      "version": "v16",
      "mode": "TYPE_B",
      "leverage": 8,
      "performance": {
        "trades": 45,
        "wins": 28,
        "losses": 17,
        "win_rate": 0.622,
        "profit_factor": 1.54,
        "net_profit": 312.40,
        "avg_win": 24.80,
        "avg_loss": -20.15
      },
      "ready_for_live": false,
      "ready_for_live_criteria": {
        "trades": { "required": 50, "current": 45, "passed": false },
        "win_rate": { "required": 0.71, "current": 0.622, "passed": false },
        "profit_factor": { "required": 1.8, "current": 1.54, "passed": false },
        "net_profit": { "required": 500, "current": 312.40, "passed": false }
      },
      "llm_iterations": 16,
      "next_optimization_in": 3
    }
  ],
  "summary": {
    "total": 10,
    "active": 8,
    "paused": 2,
    "ready_for_live": 3,
    "avg_win_rate": 0.658,
    "total_profit": 4567.80
  }
}
```

---

### 3. GET `/api/positions`
**Current open positions with live P&L**

**Response:**
```json
{
  "positions": [
    {
      "id": "pos_20241202_001",
      "strategy_id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
      "strategy_name": "Market Maker Inventory",
      "token": "BTC",
      "direction": "SELL",
      "mode": "TYPE_B",
      "leverage": 8,
      "entry": {
        "price": 96450.00,
        "time": "2024-12-02T01:10:00Z",
        "size_usd": 3000.00,
        "deviation_percent": 1.24
      },
      "current": {
        "price": 96280.00,
        "pnl_percent": 0.176,
        "pnl_usd": 42.24,
        "pnl_leveraged_percent": 1.41
      },
      "targets": {
        "tp1": { "price": 95978.50, "percent": 0.49, "status": "PENDING" },
        "tp2": { "price": 95506.50, "percent": 0.98, "status": "PENDING" },
        "tp3": { "price": 95034.50, "percent": 1.47, "status": "PENDING" }
      },
      "stop_loss": {
        "price": 97414.50,
        "percent": 1.0,
        "distance_percent": 1.18
      },
      "time_in_trade": "00:05:32",
      "status": "PROFIT"
    }
  ],
  "summary": {
    "total_positions": 12,
    "in_profit": 8,
    "in_loss": 4,
    "total_unrealized_pnl": 234.56,
    "distance_to_take_profit": 165.44,
    "take_profit_threshold": 400.00
  }
}
```

---

### 4. GET `/api/trades`
**Trade history (paginated)**

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Trades per page (default: 50)
- `strategy_id` (string): Filter by strategy
- `status` (string): Filter by WIN/LOSS/ALL

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_20241202_0847",
      "strategy_id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
      "version": "v23",
      "token": "ETH",
      "direction": "BUY",
      "mode": "TYPE_B",
      "leverage": 8,
      "entry": {
        "price": 2680.50,
        "time": "2024-12-02T00:45:00Z",
        "size_usd": 3000.00,
        "deviation_percent": 1.15
      },
      "exit": {
        "price": 2698.40,
        "time": "2024-12-02T00:52:00Z",
        "reason": "TP1_HIT"
      },
      "result": {
        "pnl_percent": 0.667,
        "pnl_usd": 160.08,
        "pnl_after_fees": 145.28,
        "duration": "00:07:00"
      },
      "status": "WIN"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total_trades": 847,
    "total_pages": 17
  },
  "summary": {
    "total_wins": 538,
    "total_losses": 309,
    "win_rate": 0.635,
    "total_profit": 4567.80
  }
}
```

---

### 5. GET `/api/portfolio`
**Portfolio P&L summary with breakdown**

**Response:**
```json
{
  "portfolio": {
    "type_a": {
      "mode": "SCALPING",
      "leverage": 3,
      "capital": {
        "initial": 17000.00,
        "current": 17845.50,
        "pnl": 845.50,
        "pnl_percent": 4.97
      },
      "trades": {
        "total": 423,
        "wins": 274,
        "losses": 149,
        "win_rate": 0.648
      },
      "open_positions": 5,
      "unrealized_pnl": 78.45
    },
    "type_b": {
      "mode": "BIG_TAKES",
      "leverage": 8,
      "capital": {
        "initial": 17000.00,
        "current": 18234.25,
        "pnl": 1234.25,
        "pnl_percent": 7.26
      },
      "trades": {
        "total": 424,
        "wins": 264,
        "losses": 160,
        "win_rate": 0.623
      },
      "open_positions": 7,
      "unrealized_pnl": 156.11
    },
    "combined": {
      "initial": 34000.00,
      "current": 36079.75,
      "realized_pnl": 2079.75,
      "unrealized_pnl": 234.56,
      "total_pnl": 2314.31,
      "pnl_percent": 6.81
    },
    "take_profit": {
      "threshold": 400.00,
      "current_unrealized": 234.56,
      "distance": 165.44,
      "percent_to_threshold": 58.64
    }
  }
}
```

---

### 6. GET `/api/ready-for-live`
**Strategies that passed 71% criteria**

**Response:**
```json
{
  "ready_strategies": [
    {
      "id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
      "name": "Market Maker Inventory Rebalancing",
      "version": "v23",
      "win_rate": 0.731,
      "profit_factor": 2.14,
      "trades": 67,
      "net_profit": 847.50,
      "llm_iterations": 23,
      "ready_since": "2024-12-02T00:30:00Z",
      "action": "POST /api/go-live/20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy"
    }
  ],
  "criteria": {
    "min_trades": 50,
    "min_win_rate": 0.71,
    "min_profit_factor": 1.8,
    "min_net_profit": 500.0
  },
  "total_ready": 3
}
```

---

## 🧠 LLM REASONING ENDPOINTS (THE BRAIN!)

### 7. GET `/api/llm-reasoning`
**Latest LLM brain activity across all strategies**

**Response:**
```json
{
  "llm_activity": [
    {
      "id": "llm_20241202_0845",
      "timestamp": "2024-12-02T00:45:00Z",
      "strategy_id": "20251124_034046_AI_Neural_Network_Predictive_Model",
      "strategy_name": "AI Neural Network Predictive",
      "from_version": "v15",
      "to_version": "v16",
      "trigger": "WIN_RATE_BELOW_TARGET",
      "current_win_rate": 0.58,
      "target_win_rate": 0.71,
      "provider": "deepseek",
      "reasoning": {
        "current_issues": [
          "Entries triggered at only 0.3% deviation - too sensitive",
          "Stop losses hit 45% of the time before TP1",
          "Missing trend confirmation leads to counter-trend trades"
        ],
        "root_causes": [
          "MIN_DEVIATION too low (0.3%) - entering weak signals",
          "STOP_DISTANCE too tight (0.5%) for 8x leverage",
          "No SMA trend filter - trading against momentum"
        ],
        "parameter_changes": {
          "MIN_DEVIATION": {
            "old": 0.003,
            "new": 0.008,
            "reason": "Require 0.8% deviation to filter weak signals"
          },
          "STOP_DISTANCE": {
            "old": 0.005,
            "new": 0.010,
            "reason": "Wider stop (1.0%) survives normal volatility at 8x"
          },
          "TP1_LEVEL": {
            "old": 0.004,
            "new": 0.006,
            "reason": "Higher TP1 (0.6%) ensures profit after costs"
          }
        },
        "logic_changes": [
          "ADD: SMA20 trend filter - only BUY when price > SMA20",
          "ADD: Volume confirmation - require above-average volume",
          "ADD: RSI filter - skip trades when RSI > 70 or < 30"
        ],
        "expected_improvement": "+8-12% win rate"
      },
      "code_diff_summary": {
        "lines_added": 45,
        "lines_removed": 12,
        "files_modified": 1
      },
      "validation_result": {
        "paper_trades": 15,
        "paper_win_rate": 0.667,
        "status": "APPLIED"
      }
    }
  ],
  "summary": {
    "total_reasoning_sessions": 156,
    "avg_improvement_per_session": 2.3,
    "most_common_issues": [
      "Entries too sensitive",
      "Stops too tight",
      "Missing trend filter"
    ]
  }
}
```

---

### 8. GET `/api/llm-reasoning/{strategy_id}`
**Full LLM reasoning history for specific strategy**

**Response:**
```json
{
  "strategy_id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
  "strategy_name": "Market Maker Inventory Rebalancing",
  "current_version": "v23",
  "total_iterations": 23,
  "reasoning_history": [
    {
      "version": "v1",
      "timestamp": "2024-11-24T06:18:12Z",
      "win_rate_before": 0.35,
      "win_rate_after": 0.42,
      "changes_summary": "Initial parameters from LLM generation"
    },
    {
      "version": "v2",
      "timestamp": "2024-11-24T08:30:00Z",
      "win_rate_before": 0.42,
      "win_rate_after": 0.48,
      "reasoning": {
        "issues": ["Stops too tight at 0.3%"],
        "fixes": ["Increased stop to 0.5%"]
      }
    },
    {
      "version": "v10",
      "timestamp": "2024-11-25T14:00:00Z",
      "win_rate_before": 0.55,
      "win_rate_after": 0.62,
      "reasoning": {
        "issues": ["Counter-trend entries failing"],
        "fixes": ["Added SMA20 trend filter"]
      }
    },
    {
      "version": "v23",
      "timestamp": "2024-12-02T00:45:00Z",
      "win_rate_before": 0.68,
      "win_rate_after": 0.731,
      "reasoning": {
        "issues": ["Small remaining leaks on news spikes"],
        "fixes": ["Added volatility regime detection"]
      }
    }
  ],
  "improvement_chart": {
    "v1": 0.35,
    "v5": 0.48,
    "v10": 0.55,
    "v15": 0.62,
    "v20": 0.68,
    "v23": 0.731
  },
  "parameters_evolution": {
    "MIN_DEVIATION": { "v1": 0.002, "v10": 0.005, "v23": 0.010 },
    "STOP_DISTANCE": { "v1": 0.003, "v10": 0.006, "v23": 0.010 },
    "TP1_LEVEL": { "v1": 0.003, "v10": 0.005, "v23": 0.008 }
  }
}
```

---

### 9. GET `/api/versions/{strategy_id}`
**All .py file versions for a strategy**

**Response:**
```json
{
  "strategy_id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
  "versions": [
    {
      "version": "v23",
      "filename": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy_v23.py",
      "path": "improved_strategies/",
      "created": "2024-12-02T00:45:00Z",
      "win_rate": 0.731,
      "status": "CURRENT",
      "is_live_ready": true
    },
    {
      "version": "v22",
      "filename": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy_v22.py",
      "path": "improved_strategies/",
      "created": "2024-12-01T22:30:00Z",
      "win_rate": 0.68,
      "status": "ARCHIVED"
    }
  ],
  "original": {
    "filename": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy.py",
    "path": "successful_strategies/",
    "created": "2024-11-24T06:18:12Z"
  },
  "total_versions": 23
}
```

---

### 10. POST `/api/go-live/{strategy_id}`
**Promote strategy to live trading**

**Request:**
```json
{
  "confirm": true,
  "capital_allocation": 5000.00,
  "max_positions": 3
}
```

**Response:**
```json
{
  "success": true,
  "message": "Strategy promoted to LIVE TRADING",
  "strategy_id": "20251124_061812_Market_Maker_Inventory_Rebalancing_Strategy",
  "version": "v23",
  "mode": "LIVE",
  "capital_allocated": 5000.00,
  "leverage": 8,
  "safety_limits": {
    "max_daily_loss": -250.00,
    "max_position_size": 750.00,
    "max_positions": 3,
    "pause_after_consecutive_losses": 3
  },
  "live_monitoring": {
    "llm_intervention_threshold": 0.65,
    "paper_validation_required": true
  }
}
```

---

### 11. GET `/api/config`
**Current system configuration**

**Response:**
```json
{
  "config": {
    "mode": "PAPER",
    "dual_mode": {
      "type_a": {
        "name": "SCALPING",
        "leverage": 3,
        "capital": 17000.00,
        "min_deviation": 0.005,
        "tp_levels": [0.004, 0.006, 0.008],
        "stop_range": [0.003, 0.008]
      },
      "type_b": {
        "name": "BIG_TAKES",
        "leverage": 8,
        "capital": 17000.00,
        "min_deviation": 0.010,
        "tp_levels": [0.008, 0.012, 0.016],
        "stop_range": [0.005, 0.015]
      }
    },
    "ready_for_live_criteria": {
      "min_trades": 50,
      "min_win_rate": 0.71,
      "min_profit_factor": 1.8,
      "min_net_profit": 500.0
    },
    "llm": {
      "primary_provider": "deepseek",
      "fallback_providers": ["openai", "anthropic"],
      "optimization_interval_cycles": 10,
      "target_win_rate": 0.71
    },
    "portfolio_take_profit": {
      "threshold": 400.00,
      "enabled": true
    },
    "safety": {
      "max_daily_loss": -500.00,
      "max_total_positions": 40,
      "pause_after_consecutive_losses": 3
    }
  }
}
```

---

## 📡 WEBSOCKET (Real-Time Updates)

### Connect: `ws://157.180.54.22:8000/ws`

**Subscribe to channels:**
```json
{
  "action": "subscribe",
  "channels": ["positions", "trades", "llm", "portfolio"]
}
```

**Real-time position update:**
```json
{
  "channel": "positions",
  "event": "update",
  "data": {
    "position_id": "pos_20241202_001",
    "token": "BTC",
    "current_price": 96280.00,
    "pnl_usd": 42.24,
    "pnl_percent": 1.41
  }
}
```

**Real-time LLM reasoning event:**
```json
{
  "channel": "llm",
  "event": "reasoning_started",
  "data": {
    "strategy_id": "20251124_034046_AI_Neural_Network_Predictive_Model",
    "trigger": "WIN_RATE_BELOW_TARGET",
    "current_win_rate": 0.58,
    "provider": "deepseek"
  }
}
```

**LLM reasoning complete:**
```json
{
  "channel": "llm",
  "event": "reasoning_complete",
  "data": {
    "strategy_id": "20251124_034046_AI_Neural_Network_Predictive_Model",
    "new_version": "v17",
    "changes_count": 5,
    "expected_improvement": "+8% win rate"
  }
}
```

---

## 🎨 FRONTEND DISPLAY SUGGESTIONS

### Dashboard Cards:
1. **Portfolio Summary** - Total P&L, Type A vs Type B comparison
2. **Active Positions** - Live updating with green/red P&L
3. **Strategy Leaderboard** - Sorted by win rate, highlight 71%+ ready
4. **LLM Brain Activity** - Real-time reasoning events (show the JSON!)

### Charts:
1. **Win Rate Evolution** - Line chart per strategy v1 → v23
2. **P&L Timeline** - Cumulative profit over time
3. **Trade Distribution** - Pie chart wins vs losses
4. **LLM Iterations** - Bar chart of improvements per strategy

### LLM Brain Panel (KEY FEATURE!):
```
┌─────────────────────────────────────────────────────────────────┐
│ 🧠 LLM BRAIN - LIVE REASONING                                   │
├─────────────────────────────────────────────────────────────────┤
│ ⏱️ 2024-12-02 01:15:00 | Strategy: AI Neural Network v15→v16   │
├─────────────────────────────────────────────────────────────────┤
│ ISSUES DETECTED:                                                │
│   • Entries at 0.3% deviation (too sensitive)                   │
│   • 45% stop-loss hit rate                                      │
│   • Missing trend confirmation                                   │
├─────────────────────────────────────────────────────────────────┤
│ CHANGES APPLIED:                                                │
│   ✏️ MIN_DEVIATION: 0.3% → 0.8%                                 │
│   ✏️ STOP_DISTANCE: 0.5% → 1.0%                                 │
│   ➕ Added SMA20 trend filter                                   │
├─────────────────────────────────────────────────────────────────┤
│ EXPECTED: +8-12% win rate improvement                           │
│ STATUS: ✅ Applied to v16                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔐 AUTHENTICATION (Future)

Currently no auth required for development. For production:

```
Header: Authorization: Bearer <jwt_token>
```

---

## 📝 ERROR RESPONSES

All endpoints return consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "STRATEGY_NOT_FOUND",
    "message": "Strategy with id '...' not found",
    "details": {}
  }
}
```

**Common error codes:**
- `STRATEGY_NOT_FOUND` - Invalid strategy ID
- `NOT_READY_FOR_LIVE` - Strategy hasn't met 71% criteria
- `INSUFFICIENT_CAPITAL` - Not enough capital for allocation
- `LLM_ERROR` - LLM provider unavailable

---

*Document created: 2024-12-02*
*API Version: 1.0*
*System: TRADEPEX ADAPTIVE*
