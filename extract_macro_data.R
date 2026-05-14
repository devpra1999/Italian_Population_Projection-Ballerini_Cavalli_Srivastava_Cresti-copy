# extract_macro_data.R
#
# Run this script ONCE from RStudio (Source button) or via:
#   Rscript extract_macro_data.R
#
# It reads estimations.RData and writes all CSV files needed by the
# Python Streamlit app into a new 'macro/' subfolder.
#
# Output: macro/df_output.csv, macro/df_debt.csv,
#         macro/mod_*_coef.csv (9 files),
#         macro/resid_*.csv    (9 files, 25 rows = 2000-2024 window each)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

load("estimations.RData")

dir.create("macro", showWarnings = FALSE)

# ---- Helpers ---------------------------------------------------------------
save_resid <- function(resid_obj, rows, name) {
  mat    <- as.matrix(resid_obj)
  window <- mat[rows, , drop = FALSE]
  df     <- cbind(year = 2000:2024, as.data.frame(window))
  write.csv(df, file.path("macro", paste0("resid_", name, ".csv")), row.names = FALSE)
  cat(sprintf("  resid_%s.csv: %d rows x %d col(s)\n", name, nrow(df), ncol(window)))
}

save_resid_vec <- function(vec, rows, name) {
  df <- data.frame(year = 2000:2024, resid = as.numeric(vec)[rows])
  write.csv(df, file.path("macro", paste0("resid_", name, ".csv")), row.names = FALSE)
  cat(sprintf("  resid_%s.csv: 25 rows x 1 col\n", name))
}

save_coef <- function(model, name) {
  coefs <- coef(model)
  df    <- data.frame(coefficient = names(coefs), value = as.numeric(coefs))
  write.csv(df, file.path("macro", paste0(name, "_coef.csv")), row.names = FALSE)
  cat(sprintf("  %s_coef.csv: %d coefficients\n", name, nrow(df)))
}

# ===========================================================================
cat("=== Data frames ===\n")
write.csv(df_output, "macro/df_output.csv", row.names = FALSE)
cat(sprintf("  df_output.csv: %d rows x %d cols\n", nrow(df_output), ncol(df_output)))

write.csv(df_debt, "macro/df_debt.csv", row.names = FALSE)
cat(sprintf("  df_debt.csv: %d rows x %d cols\n", nrow(df_debt), ncol(df_debt)))

# ===========================================================================
cat("\n=== Coefficients ===\n")
save_coef(mod_dypot_restricted_OLS, "mod_dypot_restricted_OLS")
save_coef(mod_y_restricted_OLS,     "mod_y_restricted_OLS")
save_coef(mod_pi_US,                "mod_pi_US")
save_coef(mod_pi_restricted,        "mod_pi_restricted")
save_coef(mod_r_US,                 "mod_r_US")
save_coef(mod_r_DE,                 "mod_r_DE")
save_coef(mod_r_restricted_OLS,     "mod_r_restricted_OLS")
save_coef(mod_s_social,             "mod_s_social")
save_coef(mod_s_non_social,         "mod_s_non_social")

# ===========================================================================
# Row indices match macro_simulations.R lines 74-83 exactly.
cat("\n=== Residuals (2000-2024 window) ===\n")

save_resid    (residuals(mod_y_restricted_OLS),   41:65, "dypot")      # same source as R code
save_resid    (residuals(mod_y_restricted_OLS),   41:65, "y")
save_resid_vec(residuals(mod_pi_US),              33:57, "pi_US")
save_resid    (residuals(mod_pi_restricted),      30:54, "pi")
save_resid_vec(residuals(mod_r_US),               33:57, "r_US")
save_resid_vec(residuals(mod_r_DE),                3:27, "r_DE")
save_resid    (residuals(mod_r_restricted_OLS),    1:25, "r")           # model starts at 2000
save_resid    (residuals(mod_s_social),            5:29, "s_social")
save_resid    (residuals(mod_s_non_social),        5:29, "s_non_social")

cat("\n=== Done! macro/ folder is ready. ===\n")
cat("You can now run: streamlit run final_streamlit.py\n")
