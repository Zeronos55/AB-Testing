# Step 2: the Hillstrom email data (2026-10-30)

## Dataset
- Source: Kevin Hillstrom's MineThatData E-Mail Analytics and Data Mining Challenge, March 2008
  (https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).
- Got the file by manual download and upload (the site's HTTPS certificate is broken, so the
  cloud session could not fetch it). Kept in gitignored `data/hillstrom.csv`.
- Licence: no explicit licence found. Public challenge file mirrored by several libraries.
  Do not redistribute; keep it out of git.
- 64,000 customers who bought within the previous 12 months, randomly split in three:
  no email (21,306), men's email (21,307), women's email (21,387). Matches the published sizes.
- One row = one customer. Outcomes: `visit` (visited site), `conversion` (bought), `spend`
  (dollars), measured over two weeks after the email.
- Caveat: promotional emails to existing customers, not cart reminders. No missing values.

## Raw rates (no tests yet)
| Group | Visited | Bought | Spend per person |
|---|---|---|---|
| No email | 10.62% | 0.57% | $0.65 |
| Men's email | 18.28% | 1.25% | $1.42 |
| Women's email | 15.14% | 0.88% | $1.08 |

## Prediction vs data
- User predicted a "noticeable" lift (1 to 5 points) in purchases.
- Purchases: men's +0.68 points (2.2x), women's +0.31 points (1.5x). Visits: +7.7 and +4.5 points.

## User's reading
(to fill in after discussion)
