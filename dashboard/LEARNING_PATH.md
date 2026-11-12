# Learning Power BI with this project

Goal: learn to visualise the data and the findings in Power BI Desktop (Windows). Plan agreed
2026-11-12: finish the statistics steps first, then build the dashboard in one stretch.

## Idea: rebuild charts we already trust
Each lesson rebuilds a Python chart in Power BI and gives the number it must show. If Power BI
disagrees, the data load or the formula is wrong. Same trust-but-verify habit as the rest of the project.

| Lesson | You build | Number to match |
|---|---|---|
| 1. Load and first chart | Purchase rate by email group | 0.573% none, 1.253% men's, 0.884% women's |
| 2. Distributions | Recency and past-spend charts, spend zeros | 578 spenders; 99.1% spent $0 |
| 3. Results with ranges | Lift with 95% intervals (error bars) | men's +0.68 points, 0.50 to 0.86 |
| 4. Balance | Balance dot chart | largest standardized difference 0.014 |
| 5. The dashboard | One page: slicer, KPI cards, charts, text boxes for your conclusions | |
| 6. Trust-check page | Calibration results from step 4 | see `results/04_trust_check.csv` |

## Setup (do this now, it can happen while we do statistics)
1. Install Power BI Desktop (free, Windows 10 or later): Microsoft Store, or Microsoft's Download
   Center page linked from https://learn.microsoft.com/power-bi/fundamentals/desktop-get-the-desktop
2. Get the results files: open https://github.com/Zeronos55/AB-Testing/tree/claude/elegant-hawking-34a084
   then Code, Download ZIP. Unzip it. The tables and charts are in the `results/` folder.
3. The raw data (`hillstrom.csv`) is not in GitHub. Use the file you downloaded from the blog post.

## Facts to remember
- Power BI has no built-in histogram: bin a numeric column ("New group", type Bin) and plot it as a
  column chart. Bins work on columns, not on measures.
- Error bars need columns holding the lower and upper limits (our results CSVs have them). Menu
  location depends on the Desktop version (Analytics pane or Format pane); find it on screen.
- Publishing to the online service has its own sign-in rules; check Microsoft's current page.
  For a portfolio, a `.pbix` file plus a PDF export or screenshots avoids the question.
