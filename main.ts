// GTA VI trophy rarity forecaster (Node 22, standard library only).
//
// Forecasts, for every trophy in a case, five quantiles of its Exophase
// earn rate at the case's target. The forecast is built in logit space
// from three ideas, none of which is the house baseline's persistence:
//
//  1. Counts, not rates. A trophy's earn rate is earned-count /
//     players-tracked. Both move. The program fits, per trophy, a
//     straight line of log(earned count) against log(1 + hours since
//     release) and a line of log(players tracked) against the same clock,
//     then divides the two extrapolations. When the tracked base grows
//     faster than a trophy's earned count, the rate FALLS even though
//     nobody un-earns the trophy -- the signature move of launch week,
//     when Exophase's tracked base grows by orders of magnitude while
//     each new player starts from zero trophies.
//  2. A second, independent trend: a straight line of logit(rate)
//     against log(1 + hours since release) over the recent reads (the
//     classic launch-curve shape: fast early, flattening in log time).
//     The median blends the counts/base forecast with this curve and
//     shrinks both toward the last observed rate, because a 4.5-day
//     extrapolation of a 2-day log-time slope overshoots badly if left
//     unchecked. Tier caps keep the blend honest: a platinum (100%
//     completion) cannot plausibly sit at 20% one week after launch.
//  3. An honest spread. The base spread is a random walk in logit
//     space growing with the square root of the horizon, plus the
//     trend fit's residual. On top of that, when the case's history is
//     STALE -- every read identical for hours, which on launch night
//     means Exophase has not refreshed its counts yet -- the forecast
//     is a two-component mixture: with probability p a refresh lands
//     inside the horizon and the rate drops by the log of the base
//     growth that accumulated while the page stood still; with
//     probability 1-p it does not and the rate wanders near where it
//     is. The quantiles are inverted numerically from that mixture, so
//     the lower tail stretches far down (covering the refresh) while
//     the median barely moves when p < 0.5. A narrow band around a
//     stale rate is exactly what the practice cases punish.
//
// Everything is read from the case file: trophy list, history, horizon
// and release date. No trophy is hard-coded, no network is used, and
// there is no randomness, so two runs write byte-identical output.
//
//   MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and
//   /output outside the grader's container:
//     mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp01 = (x: number): number => Math.min(100, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Least-squares line y ~ a + b*x; returns [intercept, slope]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return [my - b * mx, b];
}

/** Standard normal CDF, Abramowitz-Stegun 7.1.26 (error < 1.5e-7). */
function phi(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/**
 * Quantile of the two-component normal mixture in logit-shift space:
 * with probability `pRefresh` the shift is N(-jump, sigma), otherwise
 * N(0, sigma). Inverted by bisection on the mixture CDF; deterministic.
 */
function mixtureQuantile(
  tau: number,
  pRefresh: number,
  jump: number,
  sigma: number,
): number {
  if (pRefresh <= 0.001 || jump <= 0.001) {
    // Plain normal quantile, found by bisection too (keeps one code path).
    return normalQuantile(tau, sigma);
  }
  const cdf = (s: number): number =>
    pRefresh * phi((s + jump) / sigma) + (1 - pRefresh) * phi(s / sigma);
  let lo = -jump - 6 * sigma;
  let hi = 6 * sigma;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function normalQuantile(tau: number, sigma: number): number {
  let lo = -6 * sigma;
  let hi = 6 * sigma;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (phi(mid / sigma) < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Plausible week-one median ceiling (percent) for a trophy of this tier. */
function tierCapPct(tier: string): number {
  if (tier === 'platinum') return 2;
  if (tier === 'gold') return 30;
  if (tier === 'silver') return 50;
  return 95;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const releaseMs = Date.parse(c.game.release_date + 'T00:00:00Z');
  const checkpointMs = Date.parse(c.at);
  const horizonDays = c.horizon_minutes / 1440;
  const horizonHours = c.horizon_minutes / 60;
  const targetHours = (Date.parse(c.target) - releaseMs) / 3_600_000;
  const xTarget = Math.log(1 + Math.max(0, targetHours));

  const nTrophies = c.trophies.length;
  const last = c.history[c.history.length - 1];
  const nLast = last.players_tracked;

  // --- Per-read clock (log time since release) and staleness ----------
  const readHours: number[] = c.history.map(
    (h) => (Date.parse(h.at) - releaseMs) / 3_600_000,
  );
  const readX: number[] = readHours.map((h) => Math.log(1 + Math.max(0, h)));
  const checkpointHours = (checkpointMs - releaseMs) / 3_600_000;

  // How long the page has stood still: last read whose rates or player
  // count differ from its predecessor; if none ever changed, since the
  // first read. A page frozen for hours on launch week is a page whose
  // next refresh is overdue, not a game nobody plays.
  let staleHours = 0;
  {
    let lastChange = 0;
    for (let j = 1; j < c.history.length; j++) {
      const a = c.history[j - 1];
      const b = c.history[j];
      let changed = a.players_tracked !== b.players_tracked;
      if (!changed) {
        for (let i = 0; i < nTrophies; i++) {
          if (a.rates[i] !== b.rates[i]) {
            changed = true;
            break;
          }
        }
      }
      if (changed) lastChange = j;
    }
    const sinceMs = checkpointMs - Date.parse(c.history[lastChange].at);
    staleHours = sinceMs / 3_600_000;
  }

  // Distinct snapshots only: between Exophase refreshes the page is
  // re-read unchanged for hours. Those stale repeats carry no new
  // information, and feeding them to a trend fit drags every slope
  // toward zero (the page looks flat because it is frozen, not because
  // the game is). All trend fits below use distinct reads.
  const distinctIdx: number[] = [0];
  for (let j = 1; j < c.history.length; j++) {
    const a = c.history[j - 1];
    const b = c.history[j];
    let changed = a.players_tracked !== b.players_tracked;
    if (!changed) {
      for (let i = 0; i < nTrophies; i++) {
        if (a.rates[i] !== b.rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) distinctIdx.push(j);
  }

  // --- Tracked-base trend: log(players) ~ log(1 + hours) --------------
  const nXs: number[] = [];
  const nYs: number[] = [];
  for (const j of distinctIdx) {
    const n = c.history[j].players_tracked;
    if (n > 0) {
      nXs.push(readX[j]);
      nYs.push(Math.log(n));
    }
  }
  let baseSlopePerHour = 0; // fractional growth of the base, per hour
  let nTarget = nLast;
  if (nXs.length >= 2) {
    const [aN, bN] = ols(nXs, nYs);
    const slope = Math.min(1.5, Math.max(0, bN));
    nTarget = Math.exp(aN + slope * xTarget);
    nTarget = Math.min(nTarget, nLast * 50);
    nTarget = Math.max(nTarget, nLast);
    // Instantaneous fractional growth at the checkpoint, per hour:
    // d ln N / dh = bN / (1 + h).
    baseSlopePerHour = slope / (1 + Math.max(0, checkpointHours));
  }

  // --- Refresh mixture parameters (short horizons only) ---------------
  // Refreshes arrive roughly every few hours; the longer the page has
  // stood still, the likelier one lands inside the horizon. Model the
  // gap between refreshes as Normal(4.5h, 1.2h) and condition on having
  // waited `staleHours` already. The jump is the base growth between
  // the displayed (stale) count and the target: the log growth rate
  // times (stale time + horizon), because the displayed count was
  // already stale at the checkpoint and keeps growing to the target.
  // New players hold almost no trophies yet, so the rate is divided by
  // roughly that growth factor when the refresh lands.
  let pRefresh = 0;
  let jump = 0;
  if (horizonHours <= 24 && staleHours >= 0.75) {
    const meanGap = 4.5;
    const sdGap = 1.2;
    const waited = staleHours;
    const pWaited = phi((waited - meanGap) / sdGap);
    const pByEnd = phi((waited + horizonHours - meanGap) / sdGap);
    pRefresh = pWaited < 0.999 ? (pByEnd - pWaited) / (1 - pWaited) : 0;
    pRefresh = Math.min(0.95, Math.max(0, pRefresh));
    // Log growth rate of the base, per hour: fitted if the base was
    // moving, else a launch-night prior that decays as the launch ages
    // (the rules' own launch-night numbers imply ~0.39 log/hour).
    let g = baseSlopePerHour;
    if (nXs.length < 2 || g <= 0.001) {
      g = checkpointHours < 24 ? 0.4 : checkpointHours < 72 ? 0.1 : 0.03;
    }
    jump = g * (staleHours + horizonHours);
    jump = Math.min(1.8, Math.max(0, jump));
    if (jump < 0.05) pRefresh = 0;
  }

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nTrophies; i++) {
    const trophy = c.trophies[i];
    // Last non-null rate for this trophy.
    let rLast: number | null = null;
    for (let j = c.history.length - 1; j >= 0; j--) {
      const r = c.history[j].rates[i];
      if (r !== null && r !== undefined) {
        rLast = r;
        break;
      }
    }
    if (rLast === null) rLast = 0.01;
    const lastLogit = logit(rLast);
    const capLogit = logit(tierCapPct(trophy.tier));

    // A trophy every tracked player already holds: it stays ~held.
    // (In the launch-night data this is the one rate a refresh did not
    // move.) Keep it near 100 with a narrow band and no refresh drop.
    if (rLast >= 99.9) {
      const med = logit(99.9);
      const s = 0.15;
      const q = Z.map((z) => clamp01(round4(invLogit(med + z * s))));
      for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
      q[4] = 100; // the rate cannot exceed 100, and the observed value sits there
      rows.push({ id: trophy.id, q });
      continue;
    }

    // --- Trend fits for this trophy ----------------------------------
    const xs: number[] = [];
    const ys: number[] = [];
    const cs: number[] = [];
    const js: number[] = [];
    for (const j of distinctIdx) {
      const r = c.history[j].rates[i];
      if (r === null || r === undefined) continue;
      xs.push(readX[j]);
      ys.push(logit(r));
      // Earned counts never go below zero (a defensive clamp: a rate
      // outside [0, 100] would otherwise poison the log-count fit).
      cs.push(Math.max(0, (r * c.history[j].players_tracked) / 100));
      js.push(j);
    }
    const distinctY = new Set(ys.map((y) => y.toFixed(4))).size;
    let medianLogit = lastLogit;
    let sigmaFit = 0;
    if (xs.length >= 4 && distinctY >= 3) {
      // Recent window first (last 24h of reads), all reads if too few.
      const hLast = readHours[readHours.length - 1];
      let idx = xs.map((_, k) => k).filter((k) => readHours[js[k]] >= hLast - 24);
      if (idx.length < 3) idx = xs.map((_, k) => k);
      const wx = idx.map((k) => xs[k]);
      const wy = idx.map((k) => ys[k]);
      const [aY, bY] = ols(wx, wy);
      const slopeY = Math.min(4, Math.max(-1, bY));
      let yCurve = aY + slopeY * xTarget;
      // Fit residual (honest spread grows with misfit).
      let ss = 0;
      for (let k = 0; k < wx.length; k++) {
        const res = wy[k] - (aY + slopeY * wx[k]);
        ss += res * res;
      }
      sigmaFit = Math.sqrt(ss / Math.max(1, wx.length - 2));
      sigmaFit = Math.min(0.8, Math.max(0, sigmaFit));

      // Counts / base decomposition.
      const [aC, bC] = ols(xs, cs.map((v) => Math.log(v + 0.5)));
      const slopeC = Math.min(2, Math.max(0, bC));
      let cTarget = Math.exp(aC + slopeC * xTarget) - 0.5;
      const cLast = cs[cs.length - 1];
      cTarget = Math.max(cTarget, cLast); // earned counts never fall
      const rateCN = clampPct((100 * cTarget) / Math.max(1, nTarget));

      const combined = 0.5 * yCurve + 0.5 * logit(rateCN);
      // Shrink the extrapolation toward persistence: the further the
      // log-time extrapolation, the more shrinkage.
      const extrap = Math.max(0, xTarget - xs[xs.length - 1]);
      const wTrend = 0.65 * Math.exp(-0.25 * extrap);
      medianLogit = wTrend * combined + (1 - wTrend) * lastLogit;
      medianLogit = Math.min(medianLogit, capLogit);
      // Last-resort guard: a non-finite median must never reach the
      // output (it would serialise as null and invalidate the case).
      if (!Number.isFinite(medianLogit)) medianLogit = lastLogit;
    }

    // --- Spread -------------------------------------------------------
    // Random-walk base growing with sqrt(horizon), the fit residual,
    // and a floor for refresh/model uncertainty that is larger when
    // the history was flat (nothing was ever observed moving).
    let sigma = Math.sqrt(
      0.55 * 0.55 * horizonDays + sigmaFit * sigmaFit + 0.15 * 0.15,
    );
    if (distinctY < 3) sigma = Math.max(sigma, 0.45);
    if (trophy.tier === 'platinum') sigma += 0.25;
    else if (trophy.tier === 'gold') sigma += 0.15;
    else if (trophy.tier === 'silver') sigma += 0.05;
    sigma = Math.min(1.6, Math.max(0.3, sigma));

    const q = LEVELS.map((tau) => {
      const shift = mixtureQuantile(tau, pRefresh, jump, sigma);
      return clamp01(round4(invLogit(medianLogit + shift)));
    });
    // Guard monotonicity after rounding (mixture inversion is already
    // monotone; rounding can only tie, never invert).
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ id: trophy.id, q });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  const rows = forecastCase(c);
  const out = JSON.stringify({ trophies: rows });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), out, 'utf8');
}
