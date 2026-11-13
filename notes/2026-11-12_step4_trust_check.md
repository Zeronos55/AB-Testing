# Step 4: the trust check (2026-11-12)

Script `analysis/04_trust_check.py`, seed 20261112, 10,000 fake experiments per scenario.
Results: `results/04_trust_check.csv` (one row per scenario and method), chart `results/04_trust_check.png`.
A/A world: two groups with the same true buying rate (0.57%, about the no-email rate), so every
"difference" found is a false alarm and the true difference is 0. Hand-written z-test, Wald and
Newcombe intervals match statsmodels on 50 fake experiments. Fisher uses SciPy.
Monte Carlo noise band for 10,000 experiments: about 5% +/- 0.43 points (95% +/- 0.43).

## Observed (nominal 5% false alarms, 95% coverage)
| People per group | z-test false alarms | Fisher false alarms | Wald coverage | Newcombe coverage |
|---|---|---|---|---|
| 250 | 2.9% | 0.1% | 97.1%* | 99.3% |
| 500 | 4.5% | 1.0% | 95.5% | 97.2% |
| 1,000 | 4.6% | 2.4% | 95.4% | 95.8% |
| 2,000 | 5.0% | 2.9% | 95.0% | 95.9% |
| 5,000 | 4.5% | 3.4% | 95.5% | 95.5% |
| 10,000 | 5.3% | 4.3% | 94.7% | 94.8% |
| 21,300 (our size) | 5.0% | 4.3% | 95.0% | 95.0% |
| 500 at a 10% rate (contrast) | 5.1% | 4.1% | 94.8% | 94.9% |
Real no-email group split into random halves (10,653 each): z-test 5.6%, Fisher 3.7%, Wald and
Newcombe 94.4%. Indicative only: the 10,000 splits reuse the same people, so they are not
independent and the noise band does not apply.
*At 250 per group, 5.9% of fake experiments had zero buyers in both groups. The Wald interval is
then [0, 0], which "contains" the truth by a technicality, so its 97.1% is flattered.

## Predictions vs result
- 21,000 per group, about 5%: correct (5.0%).
- 500 per group (about 3 buyers each), "it breaks, far from 5%": not for the z-test (4.5%). It
  turns cautious, not reckless: at 250 per group it drops to 2.9%. Fisher is the one far from 5%
  (1.0% at 500), in the safe direction. The contrast row shows buyers, not people, is what matters:
  500 people at a 10% rate is calibrated.

## Reading
- At our real size the z-test and both intervals do what they promise. Fisher is valid but cautious
  (4.3%): exact tests on counts cannot hit 5% exactly.
- With few buyers the rate wobbles around 5% (discrete counts), so a gap of 0.5 points is real wobble,
  not just Monte Carlo noise.
- Limits: this checks false alarms only, not power (missing real effects). Fake data are independent
  by construction, so it cannot catch broken independence (e.g. repeated customers).

## User's reading
(to fill in after discussion)
