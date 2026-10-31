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
1. Most striking: the 1.25% purchase result in the men's-email group, which looked about 50% larger
   than the women's.
   - Clarified: 1.25% is the share of that group who bought, not an increase. Men's vs women's
     rate is 42% larger (1.25 vs 0.88); the lift over no email is 0.68 vs 0.31 points (2.2x as big).
   - Clarified: the groups are who got an email featuring men's or women's merchandise, not male
     or female customers. Columns `mens` / `womens` record what each customer bought before.
2. Hypothesis: the store's audience leans male, with some unisex/female lines. (No product info
   found.) Testable later using the `mens` / `womens` past-purchase flags.
3. Priority: purchases first (shows which products are in demand), spend per person second
   (price mix: expensive vs cheap categories), visits third (reach and attention).
   - Check: spend among buyers is about the same in all groups ($113.5 men's, $114.0 none,
     $121.9 women's), so extra spend per person comes from more buyers, not bigger baskets.
   - Limit: the data has no product column, so purchases show which email worked, not which
     products are in demand.
