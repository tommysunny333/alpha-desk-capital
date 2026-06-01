#!/usr/bin/env python3
"""
Alpha Desk Capital — Autonomous AI Trading Agent
Runs during NYSE market hours via GitHub Actions.
Calls Claude API for decisions, updates portfolio_state.json.
"""

import json
import os
import sys
from datetime import datetime, date
import pytz
import yfinance as yf
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
STARTING_CAPITAL      = 1_000_000
MAX_POSITION_SIZE     = 0.10
MAX_SWING_POSITIONS   = 5
LONG_TERM_ALLOC       = 0.60
SWING_ALLOC           = 0.30
CASH_RESERVE          = 0.10
MAX_DAILY_DRAWDOWN    = 0.02
SWING_STOP_LOSS       = 0.05
SWING_TAKE_PROFIT     = 0.12
LT_TRAILING_STOP      = 0.12
SLIPPAGE              = 0.0005
ET_ZONE               = pytz.timezone("America/New_York")

LONG_TERM_UNIVERSE = [
    "NVDA","AMD","AVGO","QCOM","AMAT","LRCX","MRVL","MU",
    "MSFT","AMZN","GOOGL","META","CRM","NOW","SNOW","DDOG",
    "CRWD","PANW","ZS","OKTA","CSCO",
    "LLY","ISRG","NVO",
    "AAPL","TSLA","ORCL","IBM","BX","V","WMT","C",
    "RKLB","CRWV","NBIS",
]

SWING_UNIVERSE = LONG_TERM_UNIVERSE + [
    "SPY","QQQ","SOXX","PLTR","COIN","UBER","COHR","DELL",
    "RGTI","AMD","BA","CVX","KO","PG","BRK-B","DIS","PYPL",
]

STATE_FILE = "portfolio_state.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now(ET_ZONE).strftime("%H:%M:%S ET")
    print(f"[{ts}] {msg}", flush=True)

def load_state() -> dict:
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state: dict):
    state["meta"]["last_updated"] = datetime.now(ET_ZONE).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    log(f"State saved → {STATE_FILE}")

def fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest prices from Yahoo Finance."""
    prices = {}
    log(f"Fetching prices for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, period="1d", interval="1m",
                           group_by="ticker", auto_adjust=True, progress=False)
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    price = float(data["Close"].dropna().iloc[-1])
                else:
                    price = float(data[ticker]["Close"].dropna().iloc[-1])
                prices[ticker] = round(price, 2)
            except Exception:
                pass
    except Exception as e:
        log(f"Price fetch error: {e}")
    log(f"Got prices for {len(prices)} tickers")
    return prices

def simulate_fill(price: float, direction: str) -> float:
    """Add slippage."""
    if direction == "BUY":
        return round(price * (1 + SLIPPAGE), 4)
    return round(price * (1 - SLIPPAGE), 4)

def portfolio_value(state: dict, prices: dict) -> float:
    cash = state["meta"]["cash"] if "cash" in state["meta"] else 0
    lt_val = sum(
        prices.get(p["ticker"], p["current_price"]) * p["shares"]
        for p in state["long_term_positions"]
    )
    sw_val = sum(
        prices.get(p["ticker"], p["current_price"]) * p["shares"]
        for p in state["swing_positions"]
    )
    return cash + lt_val + sw_val

# ── Regime Detection ──────────────────────────────────────────────────────────
def detect_regime(prices: dict) -> tuple[str, int, list[str]]:
    """Score market regime, return (regime_name, score, factors)."""
    score = 0
    factors = []

    spy_price = prices.get("SPY", 0)
    qqq_price = prices.get("QQQ", 0)
    vix_price = prices.get("^VIX", 20)

    # Fetch SMA data
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y")["Close"]
        sma50  = float(spy_hist.rolling(50).mean().iloc[-1])
        sma200 = float(spy_hist.rolling(200).mean().iloc[-1])
        spy_20d_return = (spy_price - float(spy_hist.iloc[-21])) / float(spy_hist.iloc[-21]) * 100

        spy_rsi = compute_rsi(spy_hist, 14)

        if spy_price > sma50:  score += 1; factors.append(f"SPY ${spy_price:.0f} > SMA50 ${sma50:.0f} ✓")
        else:                  score -= 1; factors.append(f"SPY ${spy_price:.0f} < SMA50 ${sma50:.0f} ✗")

        if spy_price > sma200: score += 1; factors.append(f"SPY ${spy_price:.0f} > SMA200 ${sma200:.0f} ✓")
        else:                  score -= 1; factors.append(f"SPY ${spy_price:.0f} < SMA200 ${sma200:.0f} ✗")

        if spy_rsi > 50: score += 1; factors.append(f"SPY RSI {spy_rsi:.1f} > 50 ✓")
        else:            score -= 1; factors.append(f"SPY RSI {spy_rsi:.1f} < 50 ✗")

        if spy_20d_return > 3: score += 1; factors.append(f"SPY 20d return +{spy_20d_return:.1f}% > 3% ✓")
        else:                  score -= 1; factors.append(f"SPY 20d return {spy_20d_return:.1f}% ✗")

        qqq_hist = yf.Ticker("QQQ").history(period="60d")["Close"]
        qqq_sma50 = float(qqq_hist.rolling(50).mean().iloc[-1])
        if qqq_price > qqq_sma50: score += 1; factors.append(f"QQQ ${qqq_price:.0f} > SMA50 ${qqq_sma50:.0f} ✓")
        else:                     score -= 1; factors.append(f"QQQ ${qqq_price:.0f} < SMA50 ${qqq_sma50:.0f} ✗")

        if vix_price > 25: score -= 2; factors.append(f"VIX {vix_price:.1f} > 25 ✗✗")
        else:              factors.append(f"VIX {vix_price:.1f} < 25 ✓")

    except Exception as e:
        log(f"Regime calc error: {e}")
        factors.append(f"Regime calc partial — using score {score}")

    if score >= 4:   regime = "STRONG_BULL"
    elif score >= 2: regime = "BULL"
    elif score >= -1:regime = "NEUTRAL"
    elif score >= -3:regime = "BEAR"
    else:            regime = "STRONG_BEAR"

    return regime, score, factors

def compute_rsi(prices_series, period=14) -> float:
    delta = prices_series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])

# ── Claude Decision Engine ────────────────────────────────────────────────────
def ask_claude(prompt: str) -> str:
    """Call Claude API for trading decisions."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def build_decision_prompt(
    state: dict,
    prices: dict,
    regime: str,
    regime_score: int,
    regime_factors: list[str],
    current_time: str,
) -> str:
    pv = portfolio_value(state, prices)
    cash = state["meta"].get("cash", 218462)

    lt_summary = "\n".join([
        f"  {p['ticker']}: {p['shares']} shares @ ${prices.get(p['ticker'], p['current_price']):.2f}"
        f" (entry ${p['entry_price']:.2f}, stop ${p['stop_price']:.2f},"
        f" P&L {((prices.get(p['ticker'], p['current_price'])-p['entry_price'])/p['entry_price']*100):+.1f}%)"
        for p in state["long_term_positions"]
    ])

    sw_summary = "\n".join([
        f"  {p['ticker']}: {p['shares']} shares @ ${prices.get(p['ticker'], p['current_price']):.2f}"
        f" (entry ${p['entry_price']:.2f}, stop ${p['stop_price']:.2f},"
        f" TP ${p['take_profit']:.2f}, signal: {p['signal']},"
        f" P&L {((prices.get(p['ticker'], p['current_price'])-p['entry_price'])/p['entry_price']*100):+.1f}%)"
        for p in state["swing_positions"]
    ])

    swing_slots = MAX_SWING_POSITIONS - len(state["swing_positions"])
    lt_slots = 12 - len(state["long_term_positions"])

    # Include key prices for Claude to reason about
    key_prices = {t: prices[t] for t in ["SPY","NVDA","MSFT","GOOGL","META","AVGO","CRWD","LLY","MU","MRVL","AAPL","AMZN","OKTA"] if t in prices}

    return f"""You are the AI portfolio manager for Alpha Desk Capital, a paper trading hedge fund.
Current time: {current_time} ET

PORTFOLIO STATE:
- Total Value: ${pv:,.0f}
- Cash: ${cash:,.0f} ({cash/pv*100:.1f}%)
- Starting Capital: $1,000,000
- Cumulative Return: {(pv-1000000)/1000000*100:.2f}%

MARKET REGIME: {regime} (score: {regime_score}/5)
{chr(10).join(regime_factors)}

LONG-TERM POSITIONS ({len(state['long_term_positions'])}/12 slots used):
{lt_summary or "  None"}

SWING POSITIONS ({len(state['swing_positions'])}/5 slots used):
{sw_summary or "  None"}

KEY PRICES RIGHT NOW:
{json.dumps(key_prices, indent=2)}

RULES:
- Max position size: {MAX_POSITION_SIZE*100:.0f}% of portfolio
- Swing stop loss: {SWING_STOP_LOSS*100:.0f}% below entry
- Swing take profit: {SWING_TAKE_PROFIT*100:.0f}% above entry
- LT trailing stop: {LT_TRAILING_STOP*100:.0f}% below high
- Cash reserve: minimum {CASH_RESERVE*100:.0f}%
- Never trade if circuit breaker active
- In BEAR/STRONG_BEAR: no new positions
- In NEUTRAL: reduce size 30%

TASK: You are briefing the Managing Director on this evaluation cycle. Think and write like a seasoned portfolio manager who actually runs money — not a bot reciting rules. Walk through your reasoning the way a real PM narrates a desk: what you're seeing in the tape, why it matters for THIS book, what you're doing about it, and what you're watching next.

Return ONLY a JSON object with this exact structure:
{{
  "regime_assessment": "One sharp sentence on the regime and what it means for risk appetite right now.",
  "actions": [
    {{
      "action": "BUY" | "SELL" | "HOLD" | "TRIM" | "BLOCKED",
      "ticker": "TICKER",
      "strategy": "LONG_TERM" | "SWING",
      "signal": "signal name",
      "quantity": 0,
      "price": 0.0,
      "stop_price": 0.0,
      "take_profit": 0.0,
      "reasoning": "Specific, evidence-based rationale: cite the actual indicator levels, the setup, the catalyst, and the risk/reward. E.g. 'RSI reset to 46 from 71, held SMA20 on declining volume — classic pullback-in-uptrend. Entering 5% with stop below SMA50 ($X), targeting prior high ($Y) for 2.4:1 R/R.'"
    }}
  ],
  "risk_notes": "Concrete risk flags with numbers: concentration, correlation, upcoming earnings/catalysts, positions approaching stops or TPs, drawdown proximity. Name names and levels.",
  "commentary": "This is the centrepiece — the MD reads this first. Write 4-6 sentences in the voice of a real portfolio manager talking through the cycle. Cover, in a natural narrative flow: (1) what the tape is doing and how the regime read informs your posture, (2) how the book is positioned and which names are driving P&L right now, (3) the specific reasoning behind any action you took this cycle — or, if you held, articulate WHY holding is the active decision and not just inaction, (4) the single most important risk you're monitoring with the level that would change your mind, and (5) what you're watching into the next cycle or the days ahead. Be direct, specific, and quantitative. Reference real tickers and price levels from the data above. No filler, no generic phrases like 'staying vigilant' — every sentence should carry information a PM would actually say. Vary your phrasing cycle to cycle so it never reads like a template."
}}

Only include actions where something should actually happen. If you are holding everything, return an empty actions array — but your commentary must still explain the active reasoning for standing pat (e.g. no setups cleared the bar, you're preserving dry powder ahead of a catalyst, positions are working and don't need touching). Holding is a decision, so justify it like one.
Return ONLY the JSON, no other text."""

# ── Execute Actions ───────────────────────────────────────────────────────────
def execute_actions(state: dict, actions: list[dict], prices: dict, timestamp: str):
    """Execute Claude's decisions against the portfolio state."""
    cash = state["meta"].get("cash", 218462)
    pv = portfolio_value(state, prices)
    executed = []

    for action in actions:
        ticker  = action.get("ticker", "")
        act     = action.get("action", "HOLD")
        strategy= action.get("strategy", "SWING")
        qty     = int(action.get("quantity", 0))
        price   = float(action.get("price", prices.get(ticker, 0)))
        stop    = float(action.get("stop_price", price * (1 - SWING_STOP_LOSS)))
        tp      = float(action.get("take_profit", price * (1 + SWING_TAKE_PROFIT)))
        reason  = action.get("reasoning", "")
        signal  = action.get("signal", "")

        if act == "BUY" and qty > 0 and price > 0:
            fill_price = simulate_fill(price, "BUY")
            cost = fill_price * qty
            # Risk checks
            if cost > cash - (pv * CASH_RESERVE):
                log(f"BLOCKED {ticker}: insufficient cash (need ${cost:,.0f}, available ${cash:,.0f})")
                action["action"] = "BLOCKED"
                action["reasoning"] = f"Insufficient cash. {reason}"
            elif cost / pv > MAX_POSITION_SIZE:
                log(f"BLOCKED {ticker}: position too large ({cost/pv*100:.1f}% > {MAX_POSITION_SIZE*100:.0f}%)")
                action["action"] = "BLOCKED"
            elif strategy == "SWING" and len(state["swing_positions"]) >= MAX_SWING_POSITIONS:
                log(f"BLOCKED {ticker}: swing slots full")
                action["action"] = "BLOCKED"
            else:
                cash -= cost
                position = {
                    "ticker": ticker,
                    "entry_date": timestamp[:10],
                    "entry_price": fill_price,
                    "current_price": fill_price,
                    "shares": qty,
                    "stop_price": stop,
                    "days_held": 0,
                }
                if strategy == "SWING":
                    position["take_profit"] = tp
                    position["signal"] = signal
                    state["swing_positions"].append(position)
                else:
                    state["long_term_positions"].append(position)
                log(f"EXECUTED BUY  {ticker} x{qty} @ ${fill_price:.2f} (${cost:,.0f})")
                executed.append(action)

        elif act in ("SELL", "TRIM"):
            # Find in long-term or swing
            found = False
            for book in ["long_term_positions", "swing_positions"]:
                for i, pos in enumerate(state[book]):
                    if pos["ticker"] == ticker:
                        fill_price = simulate_fill(prices.get(ticker, pos["current_price"]), "SELL")
                        sell_qty = qty if act == "TRIM" and qty > 0 else pos["shares"]
                        proceeds = fill_price * sell_qty
                        realized_pnl = (fill_price - pos["entry_price"]) * sell_qty
                        cash += proceeds
                        if sell_qty >= pos["shares"]:
                            state[book].pop(i)
                        else:
                            state[book][i]["shares"] -= sell_qty
                        log(f"EXECUTED {act} {ticker} x{sell_qty} @ ${fill_price:.2f} P&L ${realized_pnl:+,.0f}")
                        action["realized_pnl"] = round(realized_pnl, 2)
                        executed.append(action)
                        found = True
                        break
                if found:
                    break

    state["meta"]["cash"] = round(cash, 2)

    # Update current prices on all positions
    for book in ["long_term_positions", "swing_positions"]:
        for pos in state[book]:
            if pos["ticker"] in prices:
                pos["current_price"] = prices[pos["ticker"]]
            pos["days_held"] = pos.get("days_held", 0) + 0  # incremented at EOD

    return executed

# ── Stop Loss / Take Profit Checker ──────────────────────────────────────────
def check_stops_and_targets(state: dict, prices: dict) -> list[dict]:
    """Automatically trigger stops and take profits."""
    auto_actions = []
    timestamp = datetime.now(ET_ZONE).isoformat()

    for book, positions in [("swing_positions", state["swing_positions"]),
                             ("long_term_positions", state["long_term_positions"])]:
        for pos in list(positions):
            ticker = pos["ticker"]
            price = prices.get(ticker, pos["current_price"])
            stop  = pos["stop_price"]
            tp    = pos.get("take_profit")

            if price <= stop:
                log(f"STOP LOSS triggered: {ticker} price ${price:.2f} <= stop ${stop:.2f}")
                action = {"action": "SELL", "ticker": ticker, "strategy": book.replace("_positions","").upper(),
                          "price": price, "quantity": pos["shares"], "reasoning": f"Stop loss hit: ${price:.2f} <= ${stop:.2f}",
                          "signal": "STOP_LOSS"}
                auto_actions.append(action)

            elif tp and price >= tp:
                log(f"TAKE PROFIT triggered: {ticker} price ${price:.2f} >= TP ${tp:.2f}")
                action = {"action": "SELL", "ticker": ticker, "strategy": "SWING",
                          "price": price, "quantity": pos["shares"], "reasoning": f"Take profit hit: ${price:.2f} >= ${tp:.2f}",
                          "signal": "TAKE_PROFIT"}
                auto_actions.append(action)

            # Trailing stop update for LT positions
            elif book == "long_term_positions":
                new_stop = round(price * (1 - LT_TRAILING_STOP), 2)
                if new_stop > pos["stop_price"]:
                    pos["stop_price"] = new_stop

    return auto_actions

# ── Daily Snapshot ────────────────────────────────────────────────────────────
def record_daily_snapshot(state: dict, prices: dict, regime: str, daily_pnl: float):
    today = date.today().isoformat()
    pv = portfolio_value(state, prices)
    cumulative = (pv - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    spy_price = prices.get("SPY", state["meta"].get("spy_price", 756))
    spy_start = 756.48  # baseline
    spy_return = (spy_price - spy_start) / spy_start * 100

    snap = {
        "date": today,
        "portfolio_value": round(pv, 2),
        "daily_pnl": round(daily_pnl, 2),
        "cumulative_return": round(cumulative, 2),
        "spy_return": round(spy_return, 2),
        "alpha": round(cumulative - spy_return, 2),
        "regime": regime,
        "num_positions": len(state["long_term_positions"]) + len(state["swing_positions"]),
        "cash": state["meta"]["cash"],
    }

    # Update or append
    snaps = state.get("daily_snapshots", [])
    existing = next((i for i, s in enumerate(snaps) if s["date"] == today), None)
    if existing is not None:
        snaps[existing] = snap
    else:
        snaps.append(snap)

    state["daily_snapshots"] = snaps
    state["meta"]["portfolio_value"] = round(pv, 2)
    state["meta"]["cumulative_return_pct"] = round(cumulative, 2)
    state["meta"]["spy_ytd_return"] = round(spy_return, 2)
    state["meta"]["alpha_vs_spy"] = round(cumulative - spy_return, 2)
    state["meta"]["regime"] = regime
    state["meta"]["last_trading_day"] = today
    state["meta"]["total_sessions"] = len(snaps)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now_et = datetime.now(ET_ZONE)
    log(f"Alpha Desk Capital — Trading Agent starting at {now_et.strftime('%A %Y-%m-%d %H:%M ET')}")

    # Load state
    state = load_state()
    log(f"Portfolio loaded: ${state['meta']['portfolio_value']:,.0f} | "
        f"{len(state['long_term_positions'])} LT + {len(state['swing_positions'])} SW positions")

    # Collect all tickers to price
    all_tickers = list(set(
        [p["ticker"] for p in state["long_term_positions"]] +
        [p["ticker"] for p in state["swing_positions"]] +
        LONG_TERM_UNIVERSE + ["SPY","QQQ","^VIX"]
    ))

    prices = fetch_prices(all_tickers)
    if not prices:
        log("ERROR: Could not fetch any prices. Aborting.")
        sys.exit(1)

    # Regime detection
    regime, regime_score, regime_factors = detect_regime(prices)
    log(f"Regime: {regime} (score {regime_score}) — {', '.join(regime_factors[:3])}")

    # Previous portfolio value for daily P&L
    prev_value = state["meta"].get("portfolio_value", STARTING_CAPITAL)

    # Check stops/TPs first (automatic, no Claude needed)
    auto_actions = check_stops_and_targets(state, prices)
    if auto_actions:
        log(f"Auto-executing {len(auto_actions)} stop/TP actions")
        execute_actions(state, auto_actions, prices, now_et.isoformat())
        for a in auto_actions:
            a["timestamp"] = now_et.isoformat()
            state["trade_log"].append(a)

    # Circuit breaker check
    pv_now = portfolio_value(state, prices)
    daily_drawdown = (pv_now - prev_value) / prev_value if prev_value > 0 else 0
    circuit_breaker = daily_drawdown < -MAX_DAILY_DRAWDOWN
    if circuit_breaker:
        log(f"CIRCUIT BREAKER ACTIVE: daily drawdown {daily_drawdown*100:.2f}% exceeds {MAX_DAILY_DRAWDOWN*100:.0f}% limit")
        state["meta"]["circuit_breaker_active"] = True
    else:
        state["meta"]["circuit_breaker_active"] = False

    # Ask Claude for trading decisions (unless circuit breaker or strong bear)
    agent_decision = {}
    if not circuit_breaker and regime not in ("STRONG_BEAR",):
        log("Consulting Claude for trading decisions...")
        timestamp = now_et.isoformat()
        prompt = build_decision_prompt(state, prices, regime, regime_score, regime_factors, now_et.strftime("%H:%M"))
        try:
            response_text = ask_claude(prompt)
            # Strip any markdown fences
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            agent_decision = json.loads(clean)
            log(f"Claude returned {len(agent_decision.get('actions', []))} actions")
            log(f"Commentary: {agent_decision.get('commentary', '')[:200]}")

            # Execute Claude's actions
            actions = agent_decision.get("actions", [])
            if actions:
                executed = execute_actions(state, actions, prices, timestamp)
                for a in executed:
                    a["timestamp"] = timestamp
                    state["trade_log"].append(a)
            else:
                log("No trades — Claude held all positions")

        except json.JSONDecodeError as e:
            log(f"Claude response parse error: {e}")
            log(f"Raw response: {response_text[:300]}")
            agent_decision = {"commentary": "Parse error — no trades executed", "actions": []}
        except Exception as e:
            log(f"Claude API error: {e}")
            agent_decision = {"commentary": f"API error: {e}", "actions": []}
    else:
        reason = "Circuit breaker" if circuit_breaker else f"Regime {regime}"
        log(f"Skipping Claude decisions: {reason}")
        agent_decision = {"commentary": f"No new positions. {reason} prevents new entries.", "actions": []}

    # Record this decision cycle in the log
    decision_entry = {
        "timestamp": now_et.isoformat(),
        "regime": regime,
        "regime_score": regime_score,
        "portfolio_value": round(portfolio_value(state, prices), 2),
        "cash": state["meta"]["cash"],
        "actions_taken": len([a for a in agent_decision.get("actions", []) if a.get("action") not in ("HOLD","BLOCKED")]),
        "commentary": agent_decision.get("commentary", ""),
        "risk_notes": agent_decision.get("risk_notes", ""),
    }
    if "agent_decision_log" not in state:
        state["agent_decision_log"] = []
    state["agent_decision_log"].append(decision_entry)
    # Keep last 200 entries
    state["agent_decision_log"] = state["agent_decision_log"][-200:]

    # Record daily snapshot
    final_pv = portfolio_value(state, prices)
    daily_pnl = final_pv - prev_value
    record_daily_snapshot(state, prices, regime, daily_pnl)

    # Save state
    save_state(state)

    log(f"Cycle complete — Portfolio: ${final_pv:,.0f} | Daily P&L: ${daily_pnl:+,.0f} | Regime: {regime}")
    log(f"Positions: {len(state['long_term_positions'])} LT + {len(state['swing_positions'])} SW | Cash: ${state['meta']['cash']:,.0f}")

if __name__ == "__main__":
    main()
