# HavenRL — Future Roadmap & Engineering Philosophy

---

## Current State — Version 1 ✅ (Complete)

### What was built:
- 1 DQN agent trading 3 assets together (Gold ETF, Liquid ETF, Bharat Bond ETF)
- 3 actions only — Buy ALL, Sell ALL, Hold ALL (agent treats all 3 assets as one basket)
- Trained on historical data 2019-2022 (748 trading days)
- Backtested on completely unseen 2023-2024 data (490 trading days)
- Result: +3.8% return, 4.66% max drawdown, 479 trades made
- Saved model.pth (trained weights) and results.json (backtest results)
- FastAPI backend serves the pre-saved results.json
- Frontend displays portfolio chart + trade log to the user

### What the user experiences in V1:
- User opens the dashboard
- Clicks "Run Backtest"
- Sees a pre-saved portfolio performance chart for 2023-2024
- Sees the trade log (every Buy/Sell/Hold decision with date)
- Sees final metrics: Total Return %, Max Drawdown
- That's it — results are fixed, pre-saved, never change

### Limitation of V1:
- The agent treats Gold, Liquid and Bond as ONE basket
- If it says BUY — it buys all 3 at once
- If it says SELL — it sells all 3 at once
- This is too simplistic — in reality Gold and Bonds behave very differently
- Results are fixed to 2023-2024 only — user cannot change the date range
- Only 2 metrics shown (Total Return, Max Drawdown)

---

## Version 2 — 3 Specialist Agents + Dynamic Backtesting

### The core idea:
Instead of 1 agent making decisions for all 3 assets together,
we build 3 SEPARATE agents — each one is an expert in ONE asset only.

```
V1 approach (what we built):
One agent looks at all 3 assets → says "BUY everything" or "SELL everything"

V2 approach (upgrade):
Agent 1 (Gold specialist)  → looks at Gold only  → says "BUY Gold"
Agent 2 (Liquid specialist) → looks at Liquid only → says "HOLD Liquid"
Agent 3 (Bond specialist)  → looks at Bonds only  → says "SELL Bonds"
```

Now the portfolio can have mixed decisions — buy Gold while selling Bonds
while holding Liquid. This is how real portfolio managers think!

### Why specialist agents are smarter:
- Gold ETF and Bharat Bond ETF behave completely differently
- Gold spikes during geopolitical crises
- Bonds are stable and slow moving
- Liquid ETF barely moves at all — it's a cash parking vehicle
- One agent trying to learn all 3 patterns gets confused
- Three separate agents — each deeply understands its own asset

### Inspiration from production system:
This is exactly how the deep_q_24 system at Agnik works.
They have one specialist agent per Amazon campaign type (SP_TOS, SP_PP, SP_ROS etc.)
Each specialist only learns from its own campaign data.
We are applying the same pattern to trading assets.

### Technical changes needed for V2:

**Training (Colab):**
- Train 3 separate DQN agents on same historical data
- Each agent only sees its own asset's prices in the state
- Save 3 separate model files: gold_model.pth, liquid_model.pth, bond_model.pth
- Save combined results.json with all 3 agents' decisions merged

**Environment:**
- Build 3 separate TradingEnv instances — one per asset
- State for Gold agent = last 10 days of Gold prices + Gold allocation
- State for Liquid agent = last 10 days of Liquid prices + Liquid allocation
- State for Bond agent = last 10 days of Bond prices + Bond allocation

**Backend (FastAPI):**
- Load 3 model files instead of 1
- New endpoint: /backtest?start=2023-01-01&end=2024-12-31
- User passes date range as parameter
- FastAPI downloads fresh yfinance data for that range
- Runs all 3 specialist agents on that range live
- Returns fresh results — not pre-saved JSON

**Frontend:**
- Add a date range picker (start date, end date)
- User selects range → hits Run Backtest → sees fresh results
- Show individual asset performance (Gold vs Liquid vs Bond separately)
- Show combined portfolio performance

### Additional metrics shown in V2:
- Total Return % (already in V1)
- Max Drawdown (already in V1)
- Sharpe Ratio — return divided by risk (higher = better)
- Win Rate — what % of trading days were profitable
- Best Single Day gain
- Worst Single Day loss
- Trade frequency — how often each agent bought/sold vs held
- Per-asset breakdown — how much did Gold contribute vs Liquid vs Bond

### What the user experiences in V2:
- User opens dashboard
- Selects date range: e.g. "2024-01-01 to 2024-12-31"
- Clicks Run Backtest
- FastAPI downloads that year's data live from yfinance
- Runs 3 specialist agents on it
- Returns fresh results in real time
- User sees richer metrics and per-asset breakdown
- Can try different date ranges and compare results

---

## Version 3 — Live Paper Trading (Online Learning)

### The core idea:
Move from backtesting on PAST data to trading on LIVE data in real time.
But still with FAKE money — no real trades, no real risk, no SEBI compliance needed.

This is called PAPER TRADING — the agent makes real decisions on live prices
but executes them only on paper (fake portfolio).

### How it works step by step:
```
Every 5 minutes:
Step 1 → Fetch latest live prices for Gold, Liquid, Bond from Angel Broking API
Step 2 → Feed prices to 3 specialist agents as current state
Step 3 → Each agent decides: Buy / Sell / Hold
Step 4 → Update fake portfolio accordingly
Step 5 → Calculate reward (did portfolio value go up or down?)
Step 6 → Agent learns from this experience (online learning)
Step 7 → Dashboard updates with new portfolio value
Step 8 → Wait 5 minutes → repeat forever
```

### Key difference from V1 and V2:
- V1 and V2 = OFFLINE learning (train once on past data, freeze model, serve results)
- V3 = ONLINE learning (model never freezes, learns continuously from live market)

### Why this is powerful:
- Agent adapts to current market conditions — not just 2019-2022 patterns
- Dashboard shows LIVE portfolio value updating in real time
- User watches the agent make real decisions on today's actual prices
- This is exactly how deep_q_24 works at Agnik — just with fake money instead of real client money

### What is online learning?
In V1 we trained on historical data and froze the model.
Like a student who studied PYQs and appeared for one exam — done.

In V3 the model never stops learning.
Like a doctor who treats patients every day and gets better every day — never done.

New data arrives → agent acts → sees result → learns → repeat forever.

### Risks of online learning (important to know):
- Needs 24/7 monitoring — if model goes wrong, nobody to fix it at 3am
- Risk of bad/spam data corrupting the model's learning
- Risk of catastrophic forgetting — model forgets old patterns when new data dominates
- These are the same risks deep_q_24 faces in production

### Infrastructure needed for V3:
- Angel Broking API (free tier) for live NSE price feeds
- Hugging Face Spaces (free tier) as always-on server
- No broker account needed (paper trading only)
- No SEBI compliance needed (not executing real trades)
- No real money at stake

### What the user experiences in V3:
- User opens dashboard
- Sees LIVE portfolio value — updates every 5 minutes
- Watches agent make real decisions on today's actual Gold/Liquid/Bond prices
- Can see the agent learning and adapting over weeks and months
- Completely different from V1/V2 — this feels like a real trading system

---

## Version Roadmap Summary

| Version | Core Feature | Agent Type | Data | User Experience | Status |
|---|---|---|---|---|---|
| V1 | Historical backtest | 1 general agent | Fixed 2023-2024 | View pre-saved results | ✅ Done |
| V2 | Dynamic backtesting | 3 specialist agents | User picks date range | Fresh results per request | 🔄 Planned |
| V3 | Live paper trading | 3 specialist agents | Live every 5 minutes | Real time dashboard | 🔥 Future |

---

## Engineering Principles Applied

### DRY Principle & Backwards Compatibility
When upgrading V1 → V2 → V3:
- V1 users should never experience a broken dashboard
- New features added ON TOP of existing ones, not replacing them
- V1 endpoint (/results) stays working even when V2 endpoint (/backtest) is added
- Like adding a new floor to a building without shaking the foundation
- This is why software has version numbers — users on V1 still work fine

### Academia vs Real World
V1 = Academia approach
- Fixed dataset, train once, test once, done
- Like submitting an assignment

V2 = Transition
- Dynamic data, fresh results per request
- Like a deployed product that serves users

V3 = Real World approach
- Live data, online learning, never done
- Like a production system at a real company

### The Upgrade Journey (Car Price Prediction analogy)
A student builds a car price prediction model with 80/20 split and R2 score.
That's V1 thinking.

At CarDekho (real company) — new cars launch weekly, prices change daily.
The old dataset goes stale. They need online learning.
Every confirmed actual price becomes new training data. Model improves forever.
That's V3 thinking.

HavenRL follows the same journey from student project to production system.
