if(!exists("running_main")){
  rm(list = ls())
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  setwd("../")
  ### Packages-------------------------------------------------------------------
  library(dplyr)
  library(readr)
  library(pracma)
  library(panelr)
  library(ggplot2)
  library(purrr)
  library(lubridate)
  library(plm)
  library(systemfit)
  library(RColorBrewer)
  library(tidyr)
  ### Run model estimations and counterfactuals---------------------------------------------------------------------
  source("code/4_model_estimation.R")
}

df_proj <- merge(df_output,df_debt, all.x = T)
for (ctry in names(country_map)){
  df_proj[df_proj$Year>2024,paste0("sfa_",ctry)] <- 0
}
df_proj <- df_proj %>%
  fill(PTR_US, .direction = "downup")
#mod_s_restricted <- mod_s_unrestricted
rm(list = setdiff(ls(), c("df", "df_proj","mod_dypot_restricted_OLS","mod_y_restricted_OLS","mod_pi_US","mod_pi_restricted","mod_r_US","mod_r_DE","mod_r_restricted_OLS","mod_s_restricted","mod_s_social","mod_s_non_social")))
#df_proj <- df_proj %>% filter(Year > 1970)
### Set up parameters---------------------------------------------------------------------
curr_year <- 2000
curr_idx <- which(df_proj$Year==curr_year)
num_sim <- 1000
country_map <- c( "US" = "US","Germany" = "DE", "France" = "FR", "Italy" = "IT", 
                  "Netherlands" = "NL", "Spain" = "ES")

## Simulation (Optimized) -------------------------------------------------
# Coefficients
coefs_dypot <- coef(mod_dypot_restricted_OLS)
coefs_y <- coef(mod_y_restricted_OLS)
coefs_pi_US <- coef(mod_pi_US)
coefs_pi <- coef(mod_pi_restricted)
coef_r_US <- coef(mod_r_US)
coef_r_DE <- coef(mod_r_DE)
coefs_r <- coef(mod_r_restricted_OLS)
#coefs_s <- coef(mod_s_restricted)
coefs_s_social <- coef(mod_s_social)
coefs_s_non_social <- coef(mod_s_non_social)
# coefs_s_non_social["Italy_ygap_Italy"] <- 0
# coefs_s_non_social["Italy_b_lag_Italy"] <- 0
# Initialize projection matrices
country_names <- names(country_map)
n_proj <- nrow(df_proj) - curr_idx
proj_rows <- (curr_idx+1):nrow(df_proj)

matrix_template <- matrix(nrow = n_proj, ncol = num_sim)
dy_mat_list <- y_mat_list <- ygap_mat_list <- pi_mat_list <- g_mat_list <- ir_mat_list <- s_mat_list <- s_social_mat_list <- s_non_social_mat_list <- b_mat_list <- snowball_mat_list <- setNames(
  replicate(length(country_names), matrix_template, simplify = FALSE),
  country_names
)
# Error samples
e_dypot_df <- residuals(mod_y_restricted_OLS) # 1960-2100
e_y_df <- residuals(mod_y_restricted_OLS) # 1960-2100
e_pi_US_df <- residuals(mod_pi_US) #1968-2024
e_pi_df <- residuals(mod_pi_restricted) #1971-2024
#e_r_DE_df <- exp(fitted(mod_r_DE))*(exp(residuals(mod_r_DE)) - 1)# 1992-2024
e_r_US_df <- residuals(mod_r_US) #1968-2024
e_r_DE_df <- residuals(mod_r_DE) # 1992-2024
#e_r_DE_df <- residuals(mod_r_DE) # 1997-2024
e_r_df <- residuals(mod_r_restricted_OLS) # 2000-2100
e_s_df <- residuals(mod_s_restricted) # 1991-2100
e_s_social_df <- residuals(mod_s_social) # 1996-2100
e_s_non_social_df <- residuals(mod_s_non_social) # 1996-2100
e_dypot_df <- e_dypot_df[41:65,] # 2000-2024
e_y_df <- e_y_df[41:65,] # 2000-2024
e_pi_US_df <- e_pi_US_df[33:57]
e_pi_df <- e_pi_df[30:54,]
e_r_US_df <- e_r_US_df[33:57]
#e_r_DE_df <- e_r_DE_df[8:32] # 2000-2024
e_r_DE_df <- e_r_DE_df[3:27] # 2000-2024
#e_s_df <- e_s_df[10:34,] # 2000-2024
e_s_social_df <- e_s_social_df[5:29,] # 2000-2024
e_s_non_social_df <- e_s_non_social_df[5:29,] # 2000-2024
samples <- sapply(1:(num_sim*n_proj), function(x) sample(25, 1))
# Row indices
proj_rows <- (curr_idx+1):nrow(df_proj)
# Main Simulation Loop
for (n in 1:num_sim) {
  for (c in country_names) {
    #dypot_init <- mean(na.omit(df_proj[(curr_idx-20):(curr_idx+2), paste0("dypot_", c)]))
    #Make initial vectors
    dypot_vec <- dy_vec <- y_vec <- ypot <- ygap_vec <- pi_vec <- numeric(n_proj)
    ir_vec <- s_vec <- s_social_vec <- s_non_social_vec <- b_vec <- spread_vec <- b_diff_vec <- snowball_vec <- numeric(n_proj)
    # Get initial values
    y_lag <- df_proj[curr_idx, paste0("y_", c)]
    ypot_lag <- df_proj[curr_idx, paste0("ypot_", c)]
    ygap_lag <- df_proj[curr_idx, paste0("ygap_", c)]
    pi_lag <- df_proj[curr_idx, paste0("pi_",c)]
    ir_lag <- df_proj[curr_idx, paste0("IR_", c)]
    b_lag <- df_proj[[paste0("b_", c)]][curr_idx]
    for (i in 1:n_proj) {
      e_dypot <- e_dypot_df[samples[(n-1)*n_proj + i],]
      e_y <- e_y_df[samples[(n-1)*n_proj + i],]
      e_pi_US <- e_pi_US_df[samples[(n-1)*n_proj + i]]
      e_pi <- e_pi_df[samples[(n-1)*n_proj + i],]
      e_r_US <- e_r_US_df[samples[(n-1)*n_proj + i]]
      e_r_DE <- e_r_DE_df[samples[(n-1)*n_proj + i]]
      e_r <- e_r_df[samples[(n-1)*n_proj + i],]
      #e_s <- e_s_df[samples[(n-1)*n_proj + i],]
      e_s_social <- e_s_social_df[samples[(n-1)*n_proj + i],]
      e_s_non_social <- e_s_non_social_df[samples[(n-1)*n_proj + i],]
      idx <- proj_rows[i]
      # Project output growth
      dypot_vec[i] <- coefs_dypot[paste0(c, "_dn_", c)] * df_proj[idx, paste0("dn_", c)] +
        coefs_dypot[paste0(c, "_YM_", c)] * df_proj[idx, paste0("YM_", c)] +
        coefs_dypot[paste0(c, "_D_R_", c)] * df_proj[idx, paste0("D_R_", c)] + 
        e_dypot[, c]
      # Project output growth
      dy_vec[i] <- coefs_y[paste0(c, "_dypot_", c)] * dypot_vec[i] +
        coefs_y[paste0(c, "_ygap_lag_", c)] * ygap_lag +
        coefs_y[paste0(c, "_D_", c)] * df_proj[idx, paste0("D_", c)] + 
        e_y[, c]
      # Project Level of output, potential output and output gap
      y_vec[i] <- y_lag + dy_vec[i]
      ypot[i] <- ypot_lag + dypot_vec[i]
      ygap_vec[i] <- y_vec[i] - ypot[i]
      if (c=="US"){
        #Project Inflation
        pi_vec[i] <- coefs_pi_US["pi_lag_US"] * pi_lag + 
          coefs_pi_US["PTR_US"] * df_proj$PTR_US[idx] +
          e_pi_US
        #Project Cost of debt
        ir_vec[i] <- coef_r_US["(Intercept)"] +
          coef_r_US["dypot_US"] * dypot_vec[i] +
          coef_r_US["PTR_US"] * df_proj$PTR_US[idx] +
          coef_r_US["MY_US"] * df_proj$MY_US[idx] +
          e_r_US
      } else if (c=="Germany"){
        #Project Inflation
        pi_vec[i] <- coefs_pi[paste0(c, "_pi_lag_", c)] * pi_lag +
          coefs_pi[paste0(c, "_pi_US")] * pi_mat_list[["US"]][i,n] + 
          e_pi[, c]
        ir_vec[i] <- coef_r_DE["(Intercept)"] + 
          coef_r_DE["IR_US"] * ir_mat_list[["US"]][i,n] +
          coef_r_DE["lag_IR_Germany"] * ir_lag +
          e_r_DE
      } else{
        #Project Inflation
        pi_vec[i] <- coefs_pi[paste0(c, "_pi_lag_", c)] * pi_lag +
          coefs_pi[paste0(c, "_pi_US")] * pi_mat_list[["US"]][i,n] + 
          e_pi[, c]
        #Make b_diff_lag
        if (i==1){
          b_diff_lag <- b_lag - df_proj$b_Germany[curr_idx]
        } else{
          b_diff_lag <- b_lag - b_mat_list[["Germany"]][i-1,n]
        }
        #Project spread
        spread_vec[i] <- coefs_r[paste0(c, "_(Intercept)")] + 
          coefs_r[paste0(c, "_b_diff_lag_", c)] * b_diff_lag +
          e_r[, c]
        #Project cost of debt
        ir_vec[i] <- spread_vec[i] + ir_mat_list[["Germany"]][i,n]
      }
      #Surplus
      # s_vec[i] <- coefs_s[paste0(c, "_(Intercept)")] +
      #   coefs_s[paste0(c, "_ygap_", c)] * ygap_vec[i] +
      #   coefs_s[paste0(c, "_b_lag_", c)] * b_lag +
      #   #coefs_s[paste0(c, "_D_Y_", c)] * df_proj[[paste0("D_Y_", c)]][idx] +
      #   coefs_s[paste0(c, "_D_R_", c)] * df_proj[[paste0("D_R_", c)]][idx] +
      #   e_s[,c]
      s_social_vec[i] <- coefs_s_social[paste0(c, "_(Intercept)")] +
        coefs_s_social[paste0(c, "_ygap_", c)] * ygap_vec[i] +
        #coefs_s_social[paste0(c, "_b_lag_", c)] * b_lag +
        #coefs_s[paste0(c, "_D_Y_", c)] * df_proj[[paste0("D_Y_", c)]][idx] +
        coefs_s_social[paste0(c, "_D_R_", c)] * df_proj[[paste0("D_R_", c)]][idx] +
        e_s_social[,c]
      s_non_social_vec[i] <- coefs_s_non_social[paste0(c, "_(Intercept)")] +
        coefs_s_non_social[paste0(c, "_ygap_", c)] * ygap_vec[i] +
        coefs_s_non_social[paste0(c, "_b_lag_", c)] * b_lag +
        #coefs_s[paste0(c, "_D_Y_", c)] * df_proj[[paste0("D_Y_", c)]][idx] +
        #coefs_s_non_social[paste0(c, "_D_R_", c)] * df_proj[[paste0("D_R_", c)]][idx] +
        e_s_non_social[,c]
      s_vec[i] <- s_social_vec[i] + s_non_social_vec[i]
      #Debt
      b_vec[i] <- -s_vec[i] + 
        (1 + ir_vec[i])/(1 + dy_vec[i] + pi_vec[i]) * b_lag
      #Snowball Effect
      snowball_vec[i] <- (ir_vec[i] - dy_vec[i] - pi_vec[i])/(1 + dy_vec[i] + pi_vec[i]) * b_lag
      # Update lagged values for next iteration
      y_lag <- y_vec[i]
      ypot_lag <- ypot[i]
      ygap_lag <- ygap_vec[i]
      pi_lag <- pi_vec[i]
      ir_lag <- ir_vec[i]
      b_lag <- b_vec[i]
    }
    # Store results
    dy_mat_list[[c]][, n] <- dy_vec
    y_mat_list[[c]][, n] <- y_vec
    ygap_mat_list[[c]][, n] <- ygap_vec
    pi_mat_list[[c]][, n] <- pi_vec
    g_mat_list[[c]][, n] <- dy_vec + pi_vec
    ir_mat_list[[c]][, n] <- ir_vec
    s_mat_list[[c]][, n] <- s_vec
    s_social_mat_list[[c]][, n] <- s_social_vec
    s_non_social_mat_list[[c]][, n] <- s_non_social_vec
    b_mat_list[[c]][, n] <- b_vec
    snowball_mat_list[[c]][, n] <- snowball_vec
  }
}


### Plot for output simulations------------------------------------------------------
source("code/5_aux_funs_simulations.R")
blue_palette <- brewer.pal(n = 6, name = "Blues")[6:2]   # light to dark blue
names(blue_palette) <- names(country_map)[2:6]
color_palette = c(blue_palette,"US" = "red")

end_Year <- 2040
CI <- 0.9
country_names <- c("Italy","Spain","US","France","Netherlands","Germany")

dy_plot <- plot_projection("dy", df_proj, dy_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
dy_plot

pi_plot <- plot_projection("pi", df_proj, pi_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
pi_plot

g_plot <- plot_projection("g", df_proj, g_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
g_plot

ir_plot <- plot_projection("ir", df_proj, ir_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
ir_plot

b_plot <- plot_projection("b", df_proj, b_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
b_plot

s_plot <- plot_projection("s", df_proj, s_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
s_plot

s_social_plot <- plot_projection("s_social", df_proj, s_social_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
s_social_plot

s_non_social_plot <- plot_projection("s_non_social", df_proj, s_non_social_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
s_non_social_plot

snowball_plot <-plot_projection("snowball_effect", df_proj, snowball_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
snowball_plot

ygap_plot <- plot_projection("ygap", df_proj, ygap_mat_list, end_Year = end_Year, country_names = country_names, CI = CI)
ygap_plot

ggsave("plots/sim_debt_ratio_big5_oos.pdf",b_plot)
ggsave("plots/sim_real_growth_big5_oos.pdf",dy_plot)
ggsave("plots/sim_nominal_growth_big5_oos.pdf",g_plot)
ggsave("plots/sim_cost_of_debt_big5_oos.pdf",ir_plot)
ggsave("plots/sim_inflation_big5_oos.pdf",pi_plot)
ggsave("plots/sim_surplus_big5_oos.pdf",s_plot)
ggsave("plots/sim_social_surplus_big5_oos.pdf",s_social_plot)
ggsave("plots/sim_non_social_surplus_big5_oos.pdf",s_non_social_plot)
ggsave("plots/sim_snowball_effect_big5_oos.pdf",snowball_plot)
ggsave("plots/sim_output_gap_big5_oos.pdf",ygap_plot)

