# Reference fit for tests/test_fisher_scoring_bradley_terry.py.
#
# Fits the 1987 AL East `baseball` data shipped with BradleyTerry2 with and
# without a home-advantage term and writes the coefficient tables and fit
# statistics to CSV. Generated with R 4.6.1 and BradleyTerry2 1.1.3:
#
#   Rscript tests/data/baseball_btm_reference.R
library(BradleyTerry2)
data(baseball, package = "BradleyTerry2")

m1 <- BTm(cbind(home.wins, away.wins), home.team, away.team,
          data = baseball, id = "team")
baseball$home.team <- data.frame(team = baseball$home.team, at.home = 1)
baseball$away.team <- data.frame(team = baseball$away.team, at.home = 0)
m2 <- BTm(cbind(home.wins, away.wins), home.team, away.team,
          formula = ~ team + at.home, data = baseball, id = "team")

coef_table <- function(m, name) {
  s <- summary(m)$coefficients
  data.frame(model = name, term = sub("^team", "", rownames(s)),
             estimate = s[, 1], std_error = s[, 2], z_value = s[, 3],
             p_value = s[, 4], row.names = NULL)
}
fit_table <- function(m, name) {
  data.frame(model = name, deviance = deviance(m), df_residual = df.residual(m),
             aic = AIC(m), loglik = as.numeric(logLik(m)), n_params = length(coef(m)))
}
out <- "tests/data"
write.csv(rbind(coef_table(m1, "plain"), coef_table(m2, "home_advantage")),
          file.path(out, "baseball_btm_coefficients.csv"), row.names = FALSE)
write.csv(rbind(fit_table(m1, "plain"), fit_table(m2, "home_advantage")),
          file.path(out, "baseball_btm_fit.csv"), row.names = FALSE)

# Information matrices (Baltimore dropped, home-advantage model) from an
# independent glm on one row per game: expected = X'WX = solve(vcov), and the
# empirical (outer-product-of-gradients) matrix from per-game scores.
if (!requireNamespace("sandwich", quietly = TRUE))
  install.packages("sandwich", repos = "https://cloud.r-project.org", quiet = TRUE)
library(sandwich)
data(baseball, package = "BradleyTerry2")
teams <- sort(levels(baseball$home.team))
rows <- list()
for (r in seq_len(nrow(baseball))) {
  x <- setNames(rep(0, length(teams)), teams)
  x[as.character(baseball$home.team[r])] <- 1
  x[as.character(baseball$away.team[r])] <- -1
  hw <- baseball$home.wins[r]; aw <- baseball$away.wins[r]
  if (hw > 0) rows[[length(rows) + 1]] <- cbind(t(replicate(hw, x)), at.home = 1, y = 1)
  if (aw > 0) rows[[length(rows) + 1]] <- cbind(t(replicate(aw, x)), at.home = 1, y = 0)
}
games <- as.data.frame(do.call(rbind, rows))
X <- as.matrix(games[, setdiff(c(teams, "at.home"), "Baltimore")])
g <- glm(games$y ~ 0 + X, family = binomial)
colnames(X) -> nm
expected <- solve(vcov(g)); dimnames(expected) <- list(nm, nm)
empirical <- crossprod(estfun(g)); dimnames(empirical) <- list(nm, nm)
stopifnot(max(abs(coef(g) - coef(m2))) < 1e-6)
write.csv(expected, file.path(out, "baseball_glm_expected_information.csv"))
write.csv(empirical, file.path(out, "baseball_glm_empirical_information.csv"))
