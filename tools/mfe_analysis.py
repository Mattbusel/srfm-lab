import csv, statistics
from datetime import datetime

with open(r"C:\Users\Matthew\Downloads\Adaptable Light Brown Antelope_trades.csv") as f:
    rows = list(csv.DictReader(f))

pnls = [float(r["P&L"]) for r in rows]
mfes = [float(r["MFE"]) for r in rows]
maes = [float(r["MAE"]) for r in rows]

wins  = [(r,p,m) for r,p,m in zip(rows,pnls,mfes) if p > 0]
loses = [(r,p,m) for r,p,m in zip(rows,pnls,mfes) if p < 0]

win_capture = [p/m for r,p,m in wins if m > 0]

print("=== WIN ANALYSIS ===")
print(f"Total wins:          {len(wins)}")
print(f"Avg win PnL:         ${statistics.mean([p for r,p,m in wins]):,.0f}")
print(f"Avg win MFE:         ${statistics.mean([m for r,p,m in wins]):,.0f}")
print(f"Avg MFE capture:     {statistics.mean(win_capture):.1%}  (how much of peak profit we kept)")
print(f"Median MFE capture:  {statistics.median(win_capture):.1%}")
print()

actual_win_total = sum(p for r,p,m in wins)
potential_80pct  = sum(m*0.80 for r,p,m in wins if m > 0)
potential_60pct  = sum(m*0.60 for r,p,m in wins if m > 0)
print(f"Actual wins total:   ${actual_win_total:,.0f}")
print(f"At 60% MFE capture:  ${potential_60pct:,.0f}  (+${potential_60pct-actual_win_total:,.0f})")
print(f"At 80% MFE capture:  ${potential_80pct:,.0f}  (+${potential_80pct-actual_win_total:,.0f})")
print()

buckets = {"<20%":0, "20-40%":0, "40-60%":0, "60-80%":0, ">80%":0}
for c in win_capture:
    if c < 0.2:   buckets["<20%"] += 1
    elif c < 0.4: buckets["20-40%"] += 1
    elif c < 0.6: buckets["40-60%"] += 1
    elif c < 0.8: buckets["60-80%"] += 1
    else:         buckets[">80%"] += 1
print("Win MFE capture distribution:")
for k,v in buckets.items():
    bar = "#" * int(v/10)
    print(f"  {k:>8}  {v:4d} trades  {bar}")

print()
print("=== LOSS ANALYSIS ===")
print(f"Total losses:        {len(loses)}")
print(f"Avg loss PnL:        ${statistics.mean([p for r,p,m in loses]):,.0f}")
loss_mae_ratio = [abs(p)/abs(float(r["MAE"])) for r,p,m in loses if float(r["MAE"]) < 0]
if loss_mae_ratio:
    print(f"Avg MAE at exit:     {statistics.mean(loss_mae_ratio):.1%}  (1.0 = exited at max loss)")

print()
def parse_dt(s): return datetime.fromisoformat(s.replace("Z",""))
win_holds  = [(parse_dt(r["Exit Time"])-parse_dt(r["Entry Time"])).total_seconds()/3600 for r,p,m in wins]
loss_holds = [(parse_dt(r["Exit Time"])-parse_dt(r["Entry Time"])).total_seconds()/3600 for r,p,m in loses]
print("=== HOLD TIME ===")
print(f"Avg winning hold:    {statistics.mean(win_holds):.1f} hours")
print(f"Avg losing hold:     {statistics.mean(loss_holds):.1f} hours")
print(f"Median winning hold: {statistics.median(win_holds):.1f} hours")
print(f"Median losing hold:  {statistics.median(loss_holds):.1f} hours")
print()

big_wins = [(r,p,m) for r,p,m in wins if p > 100_000]
sml_wins = [(r,p,m) for r,p,m in wins if p <= 100_000]
print("=== WIN SIZE BREAKDOWN ===")
print(f"Big wins (>$100k):   {len(big_wins)}  total=${sum(p for r,p,m in big_wins):,.0f}")
print(f"Small wins (<$100k): {len(sml_wins)}  total=${sum(p for r,p,m in sml_wins):,.0f}")
if big_wins:
    big_cap = statistics.mean([p/m for r,p,m in big_wins if m > 0])
    sml_cap = statistics.mean([p/m for r,p,m in sml_wins if m > 0])
    print(f"Big win MFE capture: {big_cap:.1%}")
    print(f"Small win MFE capture:{sml_cap:.1%}")

print()
print("=== WHAT IF WE LET BIG WINS RUN ===")
# The big wins are the vol events — what if we captured 80% instead of current rate
big_actual   = sum(p for r,p,m in big_wins)
big_at_80pct = sum(m*0.80 for r,p,m in big_wins if m > 0)
total_losses = sum(p for r,p,m in loses)
new_pf = (sum(p for r,p,m in sml_wins) + big_at_80pct) / abs(total_losses)
old_pf = (actual_win_total) / abs(total_losses)
print(f"Current profit factor:           {old_pf:.3f}")
print(f"If big wins captured 80% of MFE: {new_pf:.3f}")
print(f"Additional profit from big wins: ${big_at_80pct - big_actual:,.0f}")
print()
print("=== PRACTICAL IMPLICATION ===")
print(f"The strategy closes big winners at {statistics.mean([p/m for r,p,m in big_wins if m>0]):.0%} of their peak.")
print(f"Holding them to {0.80:.0%} of peak adds ${big_at_80pct - big_actual:,.0f} total.")
avg_add = (big_at_80pct - big_actual) / len(big_wins) if big_wins else 0
print(f"That's ${avg_add:,.0f} extra per big win trade.")
print(f"Fix: trail the stop on high-conviction positions (tf_score=7, bh_mass high)")
print(f"     instead of exiting when BH flips — let it trail the BH mass decline.")
