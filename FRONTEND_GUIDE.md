# 🚀 APEX Frontend Integration Guide

Complete instructions for building a frontend dashboard to connect to the APEX trading system API.

## 📋 Table of Contents
1. [API Overview](#api-overview)
2. [Available Endpoints](#available-endpoints)
3. [Quick Start](#quick-start)
4. [Frontend Examples](#frontend-examples)
5. [Real-Time Updates](#real-time-updates)

---

## 🌐 API Overview

**Base URL:** `http://localhost:8000` (or your deployed URL)

**API Type:** REST API with JSON responses

**CORS:** Enabled for all origins (development mode)

**Authentication:** None required (add if deploying to production)

---

## 📡 Available Endpoints

### 1. **Health Check**
```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T21:00:00.000Z"
}
```

**Use Case:** Check if API is running

---

### 2. **System Status**
```
GET /api/system_status
```

**Response:**
```json
{
  "threads": {
    "strategy_discovery": "RUNNING",
    "rbi_backtest": "RUNNING",
    "champion_manager": "RUNNING",
    "whale_agent": "RUNNING",
    "sentiment_agent": "RUNNING",
    "funding_agent": "RUNNING",
    "api_server": "RUNNING"
  },
  "queues": {
    "strategy_discovery_queue": 5,
    "validated_strategy_queue": 2,
    "market_data_queue": 10
  },
  "system": {
    "uptime_seconds": 3600,
    "start_time": "2025-11-21T20:00:00.000Z",
    "iteration_count": 42
  }
}
```

**Use Case:** System monitoring dashboard

---

### 3. **Champions List**
```
GET /api/champions
```

**Response:**
```json
{
  "champions": [
    {
      "id": "champion_001",
      "status": "CHAMPION",
      "strategy_name": "RSI Divergence Strategy",
      "bankroll": 11500.00,
      "initial_bankroll": 10000.00,
      "profit_pct": 15.0,
      "total_pnl": 1500.00,
      "total_trades": 45,
      "winning_trades": 28,
      "losing_trades": 17,
      "win_rate": 62.22,
      "trades_today": 3,
      "winning_days": 8,
      "total_days": 10,
      "win_rate_days": 80.0,
      "real_trading_eligible": false,
      "created_at": "2025-11-15T10:00:00.000Z",
      "last_trade_at": "2025-11-21T20:45:00.000Z"
    }
  ],
  "summary": {
    "total_champions": 5,
    "elite": 1,
    "qualified": 2,
    "champions": 2,
    "total_bankroll": 55000.00,
    "total_profit": 8500.00,
    "total_trades": 234
  }
}
```

**Use Case:** Champions overview page

---

### 4. **Champion Detail** ⭐ NEW
```
GET /api/champion/{champion_id}
```

**Example:** `GET /api/champion/champion_001`

**Response:**
```json
{
  "champion_id": "champion_001",
  "strategy_name": "RSI Divergence Strategy",
  "status": "CHAMPION",
  
  "bankroll": {
    "current": 11500.00,
    "initial": 10000.00,
    "profit_pct": 15.0,
    "total_pnl": 1500.00,
    "current_pnl": 1200.00,
    "max_pnl": 2000.00,
    "min_pnl": -300.00
  },
  
  "statistics": {
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 62.22,
    "trades_today": 3,
    "winning_days": 8,
    "losing_days": 2,
    "total_days": 10,
    "win_rate_days": 80.0,
    "avg_profit_per_trade": 33.33,
    "biggest_win": 250.00,
    "biggest_loss": -120.00
  },
  
  "recent_trades": [
    {
      "timestamp": "2025-11-21T20:45:00.000Z",
      "action": "BUY",
      "symbol": "BTCUSDT",
      "entry_price": 43250.50,
      "actual_entry": 43271.88,
      "exit_price": 44115.00,
      "stop_loss": 42385.49,
      "take_profit": 44980.52,
      "position_size_usd": 3000.00,
      "position_size_units": 0.0693,
      "leverage": 3.0,
      "margin_required": 1000.00,
      "slippage": 1.48,
      "fees": 12.00,
      "pnl_before_fees": 58.47,
      "profit": 44.99,
      "is_winner": true,
      "bankroll_after": 11544.99,
      "confidence": 0.78
    }
  ],
  
  "daily_performance": [
    {
      "date": "2025-11-21",
      "trades": 3,
      "wins": 2,
      "pnl": 125.50,
      "win_rate": 66.67
    },
    {
      "date": "2025-11-20",
      "trades": 5,
      "wins": 3,
      "pnl": 89.25,
      "win_rate": 60.0
    }
  ],
  
  "created_at": "2025-11-15T10:00:00.000Z",
  "last_trade_at": "2025-11-21T20:45:00.000Z",
  "last_updated": "2025-11-21T21:00:00.000Z",
  "real_trading_eligible": false
}
```

**Use Case:** Individual champion dashboard page

---

### 5. **Root Endpoint**
```
GET /
```

**Response:**
```json
{
  "service": "APEX Monitoring API",
  "version": "2.0",
  "status": "running",
  "uptime_seconds": 3600
}
```

---

## 🚀 Quick Start

### Option 1: Vanilla JavaScript

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>APEX Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .champion-card { 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px;
        }
        .profit { color: green; }
        .loss { color: red; }
    </style>
</head>
<body>
    <h1>🚀 APEX Champions Dashboard</h1>
    <div id="champions"></div>

    <script>
        const API_BASE = 'http://localhost:8000';

        async function loadChampions() {
            try {
                const response = await fetch(`${API_BASE}/api/champions`);
                const data = await response.json();
                
                const championsDiv = document.getElementById('champions');
                championsDiv.innerHTML = '';
                
                data.champions.forEach(champion => {
                    const profitClass = champion.profit_pct > 0 ? 'profit' : 'loss';
                    
                    championsDiv.innerHTML += `
                        <div class="champion-card">
                            <h3>${champion.strategy_name}</h3>
                            <p><strong>Status:</strong> ${champion.status}</p>
                            <p><strong>Bankroll:</strong> $${champion.bankroll.toFixed(2)}</p>
                            <p class="${profitClass}"><strong>Profit:</strong> ${champion.profit_pct.toFixed(2)}%</p>
                            <p><strong>Win Rate:</strong> ${champion.win_rate.toFixed(2)}%</p>
                            <p><strong>Total Trades:</strong> ${champion.total_trades}</p>
                            <button onclick="viewChampion('${champion.id}')">View Details</button>
                        </div>
                    `;
                });
                
                // Show summary
                championsDiv.innerHTML += `
                    <div class="champion-card" style="background: #f0f8ff;">
                        <h3>📊 Summary</h3>
                        <p><strong>Total Champions:</strong> ${data.summary.total_champions}</p>
                        <p><strong>Total Profit:</strong> $${data.summary.total_profit.toFixed(2)}</p>
                        <p><strong>Total Trades:</strong> ${data.summary.total_trades}</p>
                    </div>
                `;
            } catch (error) {
                console.error('Error loading champions:', error);
            }
        }

        async function viewChampion(championId) {
            window.location.href = `champion.html?id=${championId}`;
        }

        // Load champions on page load
        loadChampions();
        
        // Refresh every 30 seconds
        setInterval(loadChampions, 30000);
    </script>
</body>
</html>
```

---

### Option 2: React Component

```jsx
import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

function ChampionsList() {
  const [champions, setChampions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadChampions();
    const interval = setInterval(loadChampions, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const loadChampions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/champions`);
      const data = await response.json();
      setChampions(data.champions);
      setSummary(data.summary);
      setLoading(false);
    } catch (error) {
      console.error('Error loading champions:', error);
      setLoading(false);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="champions-dashboard">
      <h1>🚀 APEX Champions</h1>
      
      {summary && (
        <div className="summary-card">
          <h2>📊 Summary</h2>
          <p>Total Champions: {summary.total_champions}</p>
          <p>Total Profit: ${summary.total_profit.toFixed(2)}</p>
          <p>Total Trades: {summary.total_trades}</p>
        </div>
      )}

      <div className="champions-grid">
        {champions.map(champion => (
          <ChampionCard key={champion.id} champion={champion} />
        ))}
      </div>
    </div>
  );
}

function ChampionCard({ champion }) {
  return (
    <div className="champion-card">
      <h3>{champion.strategy_name}</h3>
      <div className="status-badge">{champion.status}</div>
      <p>Bankroll: ${champion.bankroll.toFixed(2)}</p>
      <p className={champion.profit_pct > 0 ? 'profit' : 'loss'}>
        Profit: {champion.profit_pct.toFixed(2)}%
      </p>
      <p>Win Rate: {champion.win_rate.toFixed(2)}%</p>
      <p>Trades: {champion.total_trades}</p>
      <a href={`/champion/${champion.id}`}>View Details →</a>
    </div>
  );
}

export default ChampionsList;
```

---

### Option 3: Vue.js Component

```vue
<template>
  <div class="apex-dashboard">
    <h1>🚀 APEX Champions Dashboard</h1>
    
    <div v-if="loading">Loading...</div>
    
    <div v-else>
      <div class="summary" v-if="summary">
        <h2>📊 Summary</h2>
        <p>Total Champions: {{ summary.total_champions }}</p>
        <p>Total Profit: ${{ summary.total_profit.toFixed(2) }}</p>
        <p>Total Trades: {{ summary.total_trades }}</p>
      </div>

      <div class="champions-grid">
        <div v-for="champion in champions" 
             :key="champion.id" 
             class="champion-card"
             @click="viewChampion(champion.id)">
          <h3>{{ champion.strategy_name }}</h3>
          <span class="badge">{{ champion.status }}</span>
          <p>Bankroll: ${{ champion.bankroll.toFixed(2) }}</p>
          <p :class="champion.profit_pct > 0 ? 'profit' : 'loss'">
            Profit: {{ champion.profit_pct.toFixed(2) }}%
          </p>
          <p>Win Rate: {{ champion.win_rate.toFixed(2) }}%</p>
          <p>Trades: {{ champion.total_trades }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ApexDashboard',
  data() {
    return {
      champions: [],
      summary: null,
      loading: true,
      apiBase: 'http://localhost:8000'
    }
  },
  mounted() {
    this.loadChampions();
    setInterval(this.loadChampions, 30000); // Refresh every 30s
  },
  methods: {
    async loadChampions() {
      try {
        const response = await fetch(`${this.apiBase}/api/champions`);
        const data = await response.json();
        this.champions = data.champions;
        this.summary = data.summary;
        this.loading = false;
      } catch (error) {
        console.error('Error:', error);
        this.loading = false;
      }
    },
    viewChampion(id) {
      this.$router.push(`/champion/${id}`);
    }
  }
}
</script>

<style scoped>
.champion-card { 
  cursor: pointer; 
  border: 1px solid #ddd; 
  padding: 15px; 
  border-radius: 8px; 
}
.profit { color: green; }
.loss { color: red; }
</style>
```

---

## 📊 Champion Detail Page Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Champion Details - APEX</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .section { 
            background: #f9f9f9; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
        }
        .trade-row { 
            border-bottom: 1px solid #ddd; 
            padding: 10px 0; 
        }
        .win { color: green; }
        .loss { color: red; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>📊 Champion Detail</h1>
    <div id="champion-detail"></div>

    <script>
        const API_BASE = 'http://localhost:8000';
        const urlParams = new URLSearchParams(window.location.search);
        const championId = urlParams.get('id');

        async function loadChampionDetail() {
            try {
                const response = await fetch(`${API_BASE}/api/champion/${championId}`);
                const champion = await response.json();
                
                if (champion.error) {
                    document.getElementById('champion-detail').innerHTML = '<p>Champion not found</p>';
                    return;
                }
                
                const detailDiv = document.getElementById('champion-detail');
                
                // Header
                detailDiv.innerHTML = `
                    <h2>${champion.strategy_name}</h2>
                    <p><strong>Status:</strong> ${champion.status}</p>
                    <p><strong>Champion ID:</strong> ${champion.champion_id}</p>
                `;
                
                // Bankroll Section
                detailDiv.innerHTML += `
                    <div class="section">
                        <h3>💰 Bankroll</h3>
                        <p><strong>Current:</strong> $${champion.bankroll.current.toFixed(2)}</p>
                        <p><strong>Initial:</strong> $${champion.bankroll.initial.toFixed(2)}</p>
                        <p class="${champion.bankroll.profit_pct > 0 ? 'win' : 'loss'}">
                            <strong>Profit:</strong> ${champion.bankroll.profit_pct.toFixed(2)}%
                        </p>
                        <p><strong>Total P&L:</strong> $${champion.bankroll.total_pnl.toFixed(2)}</p>
                        <p><strong>Max P&L:</strong> $${champion.bankroll.max_pnl.toFixed(2)}</p>
                        <p><strong>Min P&L:</strong> $${champion.bankroll.min_pnl.toFixed(2)}</p>
                    </div>
                `;
                
                // Statistics Section
                detailDiv.innerHTML += `
                    <div class="section">
                        <h3>📈 Statistics</h3>
                        <p><strong>Total Trades:</strong> ${champion.statistics.total_trades}</p>
                        <p><strong>Winning Trades:</strong> ${champion.statistics.winning_trades}</p>
                        <p><strong>Losing Trades:</strong> ${champion.statistics.losing_trades}</p>
                        <p><strong>Win Rate:</strong> ${champion.statistics.win_rate.toFixed(2)}%</p>
                        <p><strong>Avg Profit per Trade:</strong> $${champion.statistics.avg_profit_per_trade.toFixed(2)}</p>
                        <p><strong>Biggest Win:</strong> $${champion.statistics.biggest_win.toFixed(2)}</p>
                        <p><strong>Biggest Loss:</strong> $${champion.statistics.biggest_loss.toFixed(2)}</p>
                    </div>
                `;
                
                // Recent Trades Table
                detailDiv.innerHTML += `
                    <div class="section">
                        <h3>📋 Recent Trades (Last 50)</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Action</th>
                                    <th>Symbol</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>Leverage</th>
                                    <th>P&L</th>
                                    <th>Fees</th>
                                    <th>Result</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${champion.recent_trades.map(trade => `
                                    <tr>
                                        <td>${new Date(trade.timestamp).toLocaleString()}</td>
                                        <td>${trade.action}</td>
                                        <td>${trade.symbol}</td>
                                        <td>$${trade.actual_entry.toFixed(2)}</td>
                                        <td>$${trade.exit_price.toFixed(2)}</td>
                                        <td>${trade.leverage}x</td>
                                        <td class="${trade.profit > 0 ? 'win' : 'loss'}">
                                            $${trade.profit.toFixed(2)}
                                        </td>
                                        <td>$${trade.fees.toFixed(2)}</td>
                                        <td>${trade.is_winner ? '✅' : '❌'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
                // Daily Performance Chart
                detailDiv.innerHTML += `
                    <div class="section">
                        <h3>📅 Daily Performance (Last 30 Days)</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Trades</th>
                                    <th>Wins</th>
                                    <th>P&L</th>
                                    <th>Win Rate</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${champion.daily_performance.map(day => `
                                    <tr>
                                        <td>${day.date}</td>
                                        <td>${day.trades}</td>
                                        <td>${day.wins}</td>
                                        <td class="${day.pnl > 0 ? 'win' : 'loss'}">
                                            $${day.pnl.toFixed(2)}
                                        </td>
                                        <td>${day.win_rate.toFixed(2)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
            } catch (error) {
                console.error('Error loading champion:', error);
                document.getElementById('champion-detail').innerHTML = 
                    '<p>Error loading champion data</p>';
            }
        }

        loadChampionDetail();
        setInterval(loadChampionDetail, 10000); // Refresh every 10s for real-time updates
    </script>
</body>
</html>
```

---

## 🔄 Real-Time Updates

### Polling Strategy (Recommended)

```javascript
// Refresh champions list every 30 seconds
setInterval(async () => {
    const response = await fetch('http://localhost:8000/api/champions');
    const data = await response.json();
    updateUI(data);
}, 30000);

// Refresh champion detail every 10 seconds (real-time trading view)
setInterval(async () => {
    const response = await fetch(`http://localhost:8000/api/champion/${championId}`);
    const data = await response.json();
    updateChampionUI(data);
}, 10000);
```

### WebSocket Alternative (Future Enhancement)

The API doesn't currently support WebSockets, but you can implement polling as shown above for real-time-like updates.

---

## 📱 Mobile-Friendly CSS

```css
/* Responsive Grid */
.champions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    padding: 20px;
}

.champion-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.champion-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: bold;
}

.status-badge.CHAMPION {
    background: #4CAF50;
    color: white;
}

.status-badge.ELITE {
    background: #FFD700;
    color: black;
}

.status-badge.QUALIFIED {
    background: #2196F3;
    color: white;
}

/* Profit/Loss Colors */
.profit { color: #4CAF50; font-weight: bold; }
.loss { color: #f44336; font-weight: bold; }

/* Responsive Tables */
@media (max-width: 768px) {
    table { font-size: 14px; }
    th, td { padding: 6px; }
}
```

---

## 🎨 Design Recommendations

### Color Scheme
- **Primary:** #2196F3 (Blue)
- **Success:** #4CAF50 (Green)
- **Danger:** #f44336 (Red)
- **Warning:** #FFD700 (Gold)
- **Background:** #f5f5f5

### Key Metrics to Display

**Champions List Page:**
- Strategy Name
- Status Badge
- Current Bankroll
- Profit %
- Win Rate
- Total Trades
- Last Trade Time

**Champion Detail Page:**
- Full Bankroll Metrics (current, profit, max/min P&L)
- Trading Statistics (win rate, avg profit, biggest win/loss)
- Recent Trades Table (with SL/TP, leverage, fees)
- Daily Performance Chart
- Real-time Trade Updates

---

## 🚨 Error Handling

```javascript
async function fetchWithErrorHandling(url) {
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        
        // Show user-friendly error
        showNotification('Unable to load data. Please try again.', 'error');
        
        return null;
    }
}
```

---

## 🔧 Testing the API

### Using cURL:

```bash
# Health check
curl http://localhost:8000/api/health

# Get all champions
curl http://localhost:8000/api/champions

# Get specific champion
curl http://localhost:8000/api/champion/champion_001

# System status
curl http://localhost:8000/api/system_status
```

### Using Postman:

1. Import endpoints as a collection
2. Set base URL variable: `{{baseUrl}} = http://localhost:8000`
3. Test each endpoint

---

## ✅ Checklist for Frontend Implementation

- [ ] Set up development environment
- [ ] Install dependencies (React/Vue/etc if needed)
- [ ] Configure API base URL
- [ ] Implement champions list page
- [ ] Implement champion detail page
- [ ] Add real-time polling (30s for list, 10s for detail)
- [ ] Style with responsive CSS
- [ ] Add error handling
- [ ] Test on mobile devices
- [ ] Add loading states
- [ ] Implement navigation between pages

---

## 🎯 Recommended Pages

1. **Dashboard** - `/`
   - System status
   - Champions summary
   - Recent activity

2. **Champions List** - `/champions`
   - Grid of all champions
   - Filterable by status
   - Sortable by profit/win rate

3. **Champion Detail** - `/champion/:id`
   - Full champion metrics
   - Trade history table
   - Daily performance chart
   - Real-time updates

4. **System Monitor** - `/system`
   - Thread status
   - Queue sizes
   - Uptime metrics

---

## 📞 Need Help?

The APEX API is running at `http://localhost:8000` once you start the system with:

```bash
python apex.py
```

All endpoints return JSON and support CORS for local development.

---

**Built with 🚀 for APEX Trading System**
