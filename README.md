# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azkhabaskara/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azkhabaskara/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01). The program (`main.ts`, Node 22, standard library only) reads one case file from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`).

## Model

The forecast is built in logit space from the case file alone — no trophy is hard-coded and nothing is fetched:

1. **Counts, not rates.** A trophy's rate is earned-count / players-tracked, and both move. Per trophy, a line of log(earned count) against log(1 + hours since release) is extrapolated and divided by the extrapolated tracked base. When the base grows faster than a trophy's earned count, the rate *falls* — the signature move of launch week, when Exophase's tracked base grows fast while each new player starts from zero trophies.
2. **A launch-curve trend.** A second line of logit(rate) against log(1 + hours since release) over the recent reads, blended with the counts/base forecast and shrunk toward the last observed rate, with tier caps (platinum, 100% completion, cannot plausibly sit at 20% a week after launch). Only *distinct* reads are fitted — between Exophase refreshes the page is re-read unchanged for hours, and those stale repeats would drag every slope toward zero.
3. **An honest spread.** A random-walk spread growing with the square root of the horizon, plus the trend fit's residual. When the history is *stale* (every read identical for hours) and the horizon is short, the forecast becomes a two-component mixture: with some probability a refresh lands inside the horizon and the rate drops by the log of the base growth that accumulated while the page stood still; otherwise it wanders near where it is. The quantiles are inverted numerically from that mixture, so the lower tail stretches far down while the median barely moves.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Local QA

On the kit's practice cases (scored against answers reconstructed from the aggregates the rules publish for the 04:30 UTC refresh read — 3,868 players tracked, 4,276 trophies earned — since the kit ships no reference answers): loss ≈ 0.48 / 0.34 (p00 / p01, counts-frozen reconstruction) vs ≈ 1.69 for the house baseline, 90% coverage 1.000, typical miss ≈ 0.26 / 0.08 points. On a synthetic final-like case (2 days of moving history, 6,510-minute horizon): loss ≈ 0.34 vs ≈ 0.41 for the baseline. Two runs on the same input are byte-identical; each case runs in ~0.2 s, far under the 8 s cap.

---

Entered on make.fun for the gta6-trophy-rarity-forecast-01 bounty.
