# Generates tests/oracle/fixtures/dm_r.json: forecast::dm.test reference values.
# Run from the repo root:  Rscript tests/oracle/fixtures/make_dm_fixtures.R
# The oracle test (tests/oracle/test_forecasting_oracle.py::test_matches_r_forecast_dm_test)
# is skipped until this file has been generated and committed.
library(forecast)
library(jsonlite)

set.seed(20261010)
cases <- list()
for (h in c(1, 3, 7)) {
  for (n in c(40, 120)) {
    # h-step errors of two forecasters: MA(h - 1) noise, different scale
    eps_a <- rnorm(n + h - 1)
    eps_b <- rnorm(n + h - 1)
    e1 <- stats::filter(eps_a, rep(1, h), sides = 1)[h:(n + h - 1)]
    e2 <- 1.1 * stats::filter(eps_b, rep(1, h), sides = 1)[h:(n + h - 1)]
    for (power in c(1, 2)) {
      r <- dm.test(e1, e2, alternative = "two.sided", h = h, power = power)
      cases[[length(cases) + 1]] <- list(
        h = h, power = power, e1 = as.numeric(e1), e2 = as.numeric(e2),
        statistic = unname(r$statistic), p_value = r$p.value
      )
    }
  }
}
write_json(list(forecast_version = as.character(packageVersion("forecast")), cases = cases),
           "tests/oracle/fixtures/dm_r.json", digits = NA, auto_unbox = TRUE)
