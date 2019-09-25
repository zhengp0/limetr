#
# robust_methods_survey.R
#
# Reed Sorensen
# July 2019
#

custom_lib_loc <- "/home/j/temp/reed/prog/R_libraries/"

library(metafor, lib.loc = custom_lib_loc)
library(robumeta, lib.loc = custom_lib_loc)
library(metaplus, lib.loc = custom_lib_loc)
library(clubSandwich, lib.loc = custom_lib_loc)
library(robustlmm, lib.loc = custom_lib_loc)
library(heavy, lib.loc = custom_lib_loc)
library(lme4)
library(dplyr)


create_dirs <- FALSE
if (create_dirs) {
  dir.create("replicates/")
  dir.create("replicates/data/")
  dir.create("replicates/results_case1/")
  dir.create("replicates/results_case2/")
  dir.create("replicates/preds/")
  dir.create("replicates/plots_case1/")
  dir.create("replicates/plots_case2/")
}


run_metareg_sim <- function(
  id, true_beta0, true_beta1, true_tau, n_studies, obs_per_study, within_study_stdev,
  outlier_offset, outlier_sd, pct_outliers, pct_outliers_assumed, save_results = TRUE, save_pdfs = FALSE) {

  dev <- FALSE

  if (dev) {
    id <- 123
    true_beta0 <- 0
    true_beta1 <- 5
    true_tau <- 6

    n_studies <- 10
    obs_per_study <- 10
    within_study_stdev <- 4
    outlier_offset <- -30
    outlier_sd <- 80
    pct_outliers <- 0.15
    pct_outliers_assumed <- 0.2
    save_results <- TRUE
    save_pdfs <- FALSE
    # df = pd.read_csv("/share/scratch/users/rsoren/misc/df2.csv")
  }

  #####
  # create synthetic data

  set.seed(id)
  library(viridis)
  library(RColorBrewer)

  # -- study-level dataset
  df_study <- data.frame(const = rep(1, n_studies)) %>%
    mutate(
      study_id = 1:nrow(.),
      sample_size = obs_per_study,
      random_intercept = rnorm(nrow(.), 0, true_tau),
      study_color = viridis(nrow(.))
    )

  # -- multiple observations per study according to 'sample_size' column
  df <- do.call("rbind", lapply(1:nrow(df_study), function(i) {
    df_study[rep(i, df_study[i, "sample_size"]), ]
  })) %>%
    mutate(
      within_study_error = rnorm(nrow(.), 0, within_study_stdev),
      x1 = runif(nrow(.), 0, 10),
      meas_value = true_beta0 + x1*true_beta1 + random_intercept + within_study_error ) %>%
    arrange(x1)

  # -- outliers
  n_above_xthres <- sum(df$x1 >= 6)
  outlier_indices <- nrow(df) - sample(0:n_above_xthres, size = round(nrow(df) * pct_outliers))
  df$outlier_status <- 0
  df$outlier_status[outlier_indices] <- 1

  df2 <- df %>%
    mutate(
      outlier_adjust = outlier_offset + sign(outlier_offset) * abs(rnorm(n = nrow(.), 0, outlier_sd)),
      y = ifelse(outlier_status, meas_value + outlier_adjust, meas_value),
      color = ifelse(outlier_status, "red", "black"),
      y_se = within_study_stdev,
      y_var = y_se^2
    )

  n_outliers_assumed <- round(nrow(df2) * pct_outliers_assumed)

  #
  # with(df2, plot(x1, y, col = color))
  # with(df2, plot(NULL, NULL, xlim = c(min(df2$x1), max(df2$x1)), ylim = c(min(df2$y), max(df2$y)), xlab = "x1", ylab = "y"))
  # for (i in unique(df2$study_id)) {
  #   dat_tmp <- filter(df2, study_id == i)
  #   with(dat_tmp, lines(x1, y, col = study_color))
  # }

  # verify that simulated data is correct
  # fit_tmp <- lme4::lmer(y ~ x1 + (1 | study_id), data = filter(df2, outlier_status == 0))
  # summary(fit_tmp)
  #
  # df2 <- df2 %>%
  #   arrange(study_id)
  #
  # write.csv(df2, "/share/scratch/users/rsoren/misc/df2_v2.csv", row.names = FALSE)

  write.csv(df2, paste0("replicates/data/df2_", id, ".csv"), row.names = FALSE)


  #####
  # case 1, known sigmas

  # # -- metafor
  time_metafor <- system.time({
    fit_metafor <- rma(yi = y, sei = y_se, mods = ~x1, data = df2)
  })
  summary(fit_metafor)

  results_metafor <- c(
    model = "metafor",
    beta0 = coef(fit_metafor)[["intrcpt"]],
    beta1 = coef(fit_metafor)[["x1"]],
    gamma = sqrt(fit_metafor$tau2),
    seconds = time_metafor[["elapsed"]]
  )

  time_metafor2 <- system.time({
    # fit_metafor2 <- metafor::robust(fit_metafor, cluster = df_sim2$x1)
    fit_metafor2 <- metafor::robust(fit_metafor, cluster = df2$study_id)
  })


  df_pred_metafor2 <- predict(fit_metafor2, newmods = df2[, "x1"])
  df2 <- df2 %>%
    mutate(
      pred_metafor2 = df_pred_metafor2$pred,
      abs_res_metafor2 = abs(y - pred_metafor2),
      outlier_metafor2 = abs_res_metafor2 >= sort(abs_res_metafor2, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  results_metafor2 <- c(
    model = "metafor with robust option",
    beta0 = coef(fit_metafor2)[["intrcpt"]],
    beta1 = coef(fit_metafor2)[["x1"]],
    gamma = sqrt(fit_metafor2$tau2),
    seconds = time_metafor[["elapsed"]] + time_metafor2[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_metafor2)
  )


  # -- robumeta
  time_robumeta <- system.time({
    fit_robumeta <- robu(y ~ x1, studynum = study_id, var.eff.size = y_var, data = df2)
  })

  df2 <- df2 %>%
    mutate(
      pred_robumeta = fit_robumeta$reg_table$b.r[1] + x1 * fit_robumeta$reg_table$b.r[2],
      abs_res_robumeta = abs(y - pred_robumeta),
      outlier_robumeta = abs_res_robumeta >= sort(abs_res_robumeta, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  results_robumeta <- c(
    model = "robumeta",
    beta0 = fit_robumeta$reg_table$b.r[1],
    beta1 = fit_robumeta$reg_table$b.r[2],
    gamma = sqrt(fit_robumeta$mod_info$tau.sq),
    seconds = time_robumeta[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_robumeta)
  )

  # -- metaplus

  run_metaplus <- TRUE
  if (run_metaplus) {


    time_metaplus <- system.time({
      fit_metaplus <- metaplus(
        yi = y, sei = y_se, mods = x1, justfit = TRUE,
        slab = study_id, random = "t-dist", data = df2
      )
    })



    saveRDS(fit_metaplus, "/home/j/temp/reed/prog/data/metareg_simulations/fit_metaplus.RDS")
    saveRDS(time_metaplus, "/home/j/temp/reed/prog/data/metareg_simulations/time_metaplus.RDS")

  } else {
    fit_metaplus <- readRDS("/home/j/temp/reed/prog/data/metareg_simulations/fit_metaplus.RDS")
    time_metaplus <- readRDS("/home/j/temp/reed/prog/data/metareg_simulations/time_metaplus.RDS")
  }

  coefs_metaplus <- fit_metaplus$fittedmodel@coef

  df2 <- df2 %>%
    mutate(
      pred_metaplus = coefs_metaplus['muhat'] + x1*coefs_metaplus['x1'],
      abs_res_metaplus = abs(y - pred_metaplus),
      outlier_metaplus = abs_res_metaplus >= sort(abs_res_metaplus, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  results_metaplus <- c(
    model = "metaplus",
    beta0 = coefs_metaplus[['muhat']],
    beta1 = coefs_metaplus[['x1']],
    gamma = sqrt(coefs_metaplus[['tau2']]),
    seconds = time_metaplus[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_metaplus)
  )


  # -- LimeTR (results from Python)
  # will overwrite with mean of results from the replicates

  coefs_limetr <- c(beta0 = -2.87081199, beta1 = 5.5484884, gamma = 3.99396919)
  time_limetr <- 0.2566516399383545
  true_outliers_detected_limetr <- 15
  df2 <- df2 %>%
    mutate(pred_limetr = coefs_limetr[1] + x1*coefs_limetr[2])


  results_limetr <- c(
    model = "LimeTR",
    beta0 = coefs_limetr[["beta0"]],
    beta1 = coefs_limetr[["beta1"]],
    gamma = coefs_limetr[["gamma"]],
    seconds = time_limetr,
    true_outliers_detected = true_outliers_detected_limetr
  )

  if (save_pdfs) pdf(paste0("replicates/plots_case1/case1_results_", id, ".pdf"))

  # with(df2, plot(x1, y, col = color, xlab = "", ylab = "", pch = df2$outlier_pch_metafor2))
  with(df2, plot(x1, y, col = color, xlab = "", ylab = ""))
  with(df2, abline(a = true_beta0, b = true_beta1, lwd = 3))
  with(df2, lines(x1, pred_limetr, lty = 2, lwd = 2))
  with(df2, lines(x1, pred_robumeta, col = "red", lty = 3, lwd = 2))
  with(df2, lines(x1, pred_metafor2, col = "blue", lty = 4, lwd = 2))
  with(df2, lines(x1, pred_metaplus, col = "darkorange", lty = 5, lwd = 2))

  legend(
    "bottomleft",
    legend = c("Inlier", "Outlier", "beta = [0,5]", "LimeTr", "robumeta", "metafor", "metaplus"),
    col = c("black", "red", "black", "black", "red", "blue", "darkorange"),
    pch = c(1, 1, NA, NA, NA, NA, NA),
    lty = c(NA, NA, 1, 2, 3, 4, 5),
    lwd = c(NA, NA, 3, 2, 2, 2, 2)
  )


  if (save_pdfs) dev.off()

  # with(filter(df_tmp, w < 1), points(x1, meas_value))



  # -- true values
  results_true <- c(
    model = "True values",
    beta0 = true_beta0,
    beta1 = true_beta1,
    gamma = true_tau,
    seconds = NA,
    true_outliers_detected = round(nrow(df2) * pct_outliers)
  )


  results_case1 <- data.frame(stringsAsFactors = FALSE,
    bind_rows(results_true, results_limetr, results_metafor2, results_robumeta, results_metaplus)
  )

  results_case1[2:ncol(results_case1)] <- lapply(results_case1[2:ncol(results_case1)], function(x) {
    round(as.numeric(x), digits = 3)
  })

  results_case1_2 <- results_case1 %>%
    mutate(
      sample_size = nrow(df2),
      re_beta0 = beta0 - true_beta0,
      re_beta1 = beta1 - true_beta1,
      re_gamma = gamma - true_tau,
      tpf = true_outliers_detected / round(nrow(df2) * pct_outliers),
      fp_num = round(nrow(df2) * pct_outliers_assumed) - true_outliers_detected,
      fpf = fp_num / round(nrow(df2) * (1-pct_outliers))
    )

  results_case1_2


  write.csv(results_case1_2, paste0("replicates/results_case1/results_case1_", id, ".csv"), row.names = FALSE)

  ###

  df_preds_case1 <- df2 %>%
    mutate(
      res_metafor2 = y - pred_metafor2,
      res_robumeta = y - pred_robumeta,
      res_mrbrt = y - pred_limetr,
      res_metaplus = y - pred_metaplus
    )

  write.csv(df_preds_case1, paste0("replicates/preds/df_preds_case1_", id, ".csv"), row.names = FALSE)



  #####
  # case 2

  # -- lme4
  time_lme4 <- system.time({
    fit_lme4 <- lme4::lmer(y ~ x1 + (1 | study_id), data = df2)
  })

  run_lme4_validation_tests <- FALSE
  if (run_lme4_validation_tests) {

    # check estimated values without outliers
    dat_inliers <- filter(df2, outlier_status == 0)
    fit_lme4_tmp <- lme4::lmer(y ~ x1 + (1 | study_id), data = dat_inliers)
    summary(fit_lme4_tmp)

    # check alternative method for calculating sigma;
    #  will use for 'heavy' package which doesn't report sigma
    df_pred_lme4 <- predict(fit_lme4, re.form = NA)
    res_lme4 <- df2$y - df_pred_lme4
    sigma_lme4 <- sqrt(sd(res_lme4)^2 - sd(ranef(fit_lme4)[[1]][,1])^2)
    summary(fit_lme4) # 'sigma_lme4' approximates the lme4 sigma value
  }

  tmp_summary <- summary(fit_lme4)
  tmp_coefs <- as.numeric(summary(fit_lme4)$coefficients[,1])
  df_pred_lme4 <- tmp_coefs[1] + df2$x1 * tmp_coefs[2]

  df2 <- df2 %>%
    mutate(
      pred_lme4 = as.numeric(df_pred_lme4),
      abs_res_lme4 = abs(y - pred_lme4),
      outlier_lme4 = abs_res_lme4 >= sort(abs_res_lme4, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  results_lme4 <- c(
    model = "lme4",
    beta0 = tmp_coefs[1],
    beta1 = tmp_coefs[2],
    gamma = attr(tmp_summary$varcor$study_id, "stddev")[[1]],
    sigma = tmp_summary$sigma,
    seconds = time_lme4[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_lme4)
  )




  # -- robustlmm
  time_robustlmm <- system.time({
    fit_robustlmm <- robustlmm::rlmer(y ~ x1 + (1|study_id), data = df2)
  })
  tmp_coefs <- as.numeric(getInfo(fit_robustlmm)$coef)
  df_pred_robustlmm <- tmp_coefs[1] + df2$x1 * tmp_coefs[2]

  tmp_robustlmm <- summary(fit_robustlmm)

  # df_pred_robustlmm <- predict(
  #   fit_robustlmm,
  #   newdata = select(df2, x1, study_id) %>% mutate(study_id = 9999),
  #   allow.new.levels = TRUE
  # )

  df2 <- df2 %>%
    mutate(
      pred_robustlmm = as.numeric(df_pred_robustlmm),
      abs_res_robustlmm = abs(y - pred_robustlmm),
      outlier_robustlmm = abs_res_robustlmm >= sort(abs_res_robustlmm, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  results_robustlmm <- c(
    model = "robustlmm",
    beta0 = tmp_coefs[1],
    beta1 = tmp_coefs[2],
    gamma = sqrt(tmp_robustlmm$varcor$study_id),
    sigma = tmp_robustlmm$sigma,
    seconds = time_robustlmm[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_robustlmm)
  )

  # -- heavy

  time_heavy <- system.time({
    fit_heavy <- heavyLme(
      fixed = y ~ x1,
      random = ~ const,
      groups = ~ study_id,
      data = df2
    )
  })

  # df_pred_heavy <- fit_heavy$Fitted$marginal
  tmp_coefs2 <- as.numeric(coef(fit_heavy))
  df_pred_heavy <- tmp_coefs2[1] + df2$x1 * tmp_coefs2[2]

  df2 <- df2 %>%
    mutate(
      pred_heavy = as.numeric(df_pred_heavy),
      abs_res_heavy = abs(y - pred_heavy),
      outlier_heavy = abs_res_heavy >= sort(abs_res_heavy, decreasing=T)[round(nrow(df2) * pct_outliers_assumed)]
    )

  sigma_heavy <- sqrt(sd(df2$y - df2$pred_heavy)^2 - sd(fit_heavy$ranef[,1])^2)

  results_heavy <- c(
    model = "heavy",
    beta0 = tmp_coefs2[1],
    beta1 = tmp_coefs2[2],
    gamma = sd(fit_heavy$ranef[,1]),
    sigma = sigma_heavy,
    seconds = time_heavy[["elapsed"]],
    true_outliers_detected = sum(df2$outlier_status & df2$outlier_heavy)
  )


  # -- limetr
  coefs_limetr2 <- c(beta0 = 1.96045627, beta1 = 4.49915655, gamma = 0.03291021)
  time_limetr2 <- 0.3465437889099121
  true_outliers_detected_limetr2 <- 15
  df2 <- df2 %>%
    mutate(pred_limetr2 = coefs_limetr2[1] + x1*coefs_limetr2[2])

  results_limetr2 <- c(
    model = "LimeTR",
    beta0 = coefs_limetr2[["beta0"]],
    beta1 = coefs_limetr2[["beta1"]],
    gamma = coefs_limetr2[["gamma"]],
    seconds = time_limetr2,
    true_outliers_detected = true_outliers_detected_limetr2
  )


  if (save_pdfs) pdf(paste0("replicates/plots_case2/case2_results_", id, ".pdf"))

  df2 <- arrange(df2, x1)
  with(df2, plot(x1, y, col = color, xlab = "", ylab = ""))
  with(df2, abline(a = true_beta0, b = true_beta1, lwd = 3))
  with(df2, lines(x1, pred_limetr2, lty = 2, lwd = 2))
  with(df2, lines(x1, pred_heavy, col = "darkorange", lty = 3, lwd = 3))
  with(df2, lines(x1, pred_robustlmm, col = "blue", lty = 4, lwd = 2))
  with(df2, lines(x1, pred_lme4, col = "darkred", lty = 5, lwd = 2))

  legend(
    "bottomleft",
    legend = c("Inlier", "Outlier", "beta = [0,5]", "LimeTR", "heavy", "robustlmm", "lme4"),
    col = c("black", "red", "black", "black", "darkorange", "blue", "darkred"),
    pch = c(1, 1, NA, NA, NA, NA, NA),
    lty = c(NA, NA, 1, 2, 3, 4, 5),
    lwd = c(NA, NA, 3, 2, 3, 2, 2)
  )


  if (save_pdfs) dev.off()

  # with(filter(df_tmp, w < 1), points(x1, meas_value))

  # -- true values
  results_true <- c(
    model = "True values",
    beta0 = true_beta0,
    beta1 = true_beta1,
    gamma = true_tau,
    sigma = within_study_stdev,
    seconds = NA
  )


  results_case2 <- data.frame(stringsAsFactors = FALSE,
    bind_rows(results_true, results_limetr2, results_robustlmm, results_heavy, results_lme4)
  )

  results_case2[2:ncol(results_case2)] <- lapply(results_case2[2:ncol(results_case2)], function(x) {
    round(as.numeric(x), digits = 3)
  })

  results_case2_2 <- results_case2 %>%
    mutate(
      sample_size = nrow(df2),
      re_beta0 = beta0 - true_beta0,
      re_beta1 = beta1 - true_beta1,
      re_gamma = gamma - true_tau,
      re_sigma = sigma - within_study_stdev,
      tpf = true_outliers_detected / round(nrow(df2) * pct_outliers),
      fp_num = round(nrow(df2) * pct_outliers_assumed) - true_outliers_detected,
      fpf = fp_num / round(nrow(df2) * (1-pct_outliers))
    )

  write.csv(results_case2_2, paste0("replicates/results_case2/results_case2_", id, ".csv"), row.names = FALSE)

  df_preds_case2 <- df2 %>%
    mutate(
      # res_lme4 = meas_value - pred_lme4,
      res_robustlmm = meas_value - pred_robustlmm,
      res_heavy = meas_value - pred_heavy
    )

  write.csv(df_preds_case2, paste0("replicates/preds/df_preds_case2.csv", row.names = FALSE))

}


library(parallel)


# for (i in 0:29) {
mclapply(0:29, function(i) {

  run_metareg_sim(
    id = i,
    true_beta0 = 0,
    true_beta1 = 5,
    true_tau = 6,
    n_studies = 10,
    obs_per_study = 10,
    within_study_stdev = 4,
    outlier_offset = -30,
    outlier_sd = 80,
    pct_outliers = 0.15,
    pct_outliers_assumed = 0.2,
    save_results = TRUE,
    save_pdfs = FALSE
  )

}, mc.cores = 30)


#####

# aggregate replicates of case 1 simulation

dat_path_case1 <- "replicates/results_case1/"
results_files_case1 <- list.files(dat_path_case1)

dat_agg_case1 <- lapply(0:29, function(x) {
  tmp <- read.csv(paste0(dat_path_case1, "/results_case1_", x, ".csv")) %>%
    mutate(replicate_id = x)
})

df_agg_case1 <- do.call("rbind", dat_agg_case1)

df_agg_case1_mean <- df_agg_case1 %>%
  group_by(model) %>%
  summarize_all(mean) %>%
  mutate(case = "case1")



write.csv(df_agg_case1, "replicates/df_agg_case1.csv", row.names = FALSE)

# aggregate replicates of case 1 simulation

dat_path_case2 <- "replicates/results_case2/"
results_files_case2 <- list.files(dat_path_case2)
model_names_case2 <- as.character(read.csv(paste0(dat_path_case2, results_files_case2[1]))[, "model"])

dat_agg_case2 <- lapply(0:29, function(x) {
  tmp <- read.csv(paste0(dat_path_case2, "/results_case2_", x, ".csv")) %>%
    mutate(replicate_id = x)
})

df_agg_case2 <- do.call("rbind", dat_agg_case2)

df_agg_case2_mean <- df_agg_case2 %>%
  group_by(model) %>%
  summarize_all(mean) %>%
  mutate(case = "case2")

write.csv(df_agg_case2, "replicates/df_agg_case2.csv", row.names = FALSE)



# combine with results from LimeTR (ran LimeTR in Python)
# -- case 1
dat_path_case1_limetr <- "replicates/results_case1_limetr/"
results_files_case1_limetr <- list.files(dat_path_case1_limetr)
model_names_case1_limetr <- as.character(
  read.csv(paste0(dat_path_case1_limetr, results_files_case1_limetr[1]))[, "model"]
)

dat_agg_case1_limetr <- lapply(0:29, function(x) {
  tmp <- read.csv(paste0(dat_path_case1_limetr, "/results_case1_limetr_", x, ".csv")) %>%
    mutate(replicate_id = x)
})

df_agg_case1_limetr <- do.call("rbind", dat_agg_case1_limetr)

df_agg_case1_limetr_mean <- df_agg_case1_limetr %>%
  group_by(model) %>%
  summarize_all(mean) %>%
  mutate(case = "case1")

# -- case 2
dat_path_case2_limetr <- "replicates/results_case2_limetr/"
results_files_case2_limetr <- list.files(dat_path_case2_limetr)
model_names_case2_limetr <- as.character(
  read.csv(paste0(dat_path_case2_limetr, results_files_case2_limetr[1]))[, "model"]
)

dat_agg_case2_limetr <- lapply(0:29, function(x) {
  tmp <- read.csv(paste0(dat_path_case2_limetr, "/results_case2_limetr_", x, ".csv")) %>%
    mutate(replicate_id = x)
})

df_agg_case2_limetr <- do.call("rbind", dat_agg_case2_limetr)

df_agg_case2_limetr_mean <- df_agg_case2_limetr %>%
  group_by(model) %>%
  summarize_all(mean) %>%
  mutate(case = "case2")


#####
# final results


df_final_nonlimetr <- bind_rows(
  list(df_agg_case1_mean, df_agg_case2_mean) ) %>%
  filter(model != "LimeTR")

df_final_limetr <- bind_rows(
  list(df_agg_case1_limetr_mean, df_agg_case2_limetr_mean)
)

df_final <- bind_rows(df_final_nonlimetr, df_final_limetr) %>%
  arrange(case)


nms <- names(df_final)
nms2 <- nms[!nms == "case"]
nms3 <- c(nms2[1], "case", nms2[2:length(nms2)])

df_final2 <- df_final[nms3]

df_final2[, -c(1,2)] <- round(df_final2[, -c(1,2)], digits = 2)


write.csv(df_final2, "df_final.csv", row.names = FALSE)










