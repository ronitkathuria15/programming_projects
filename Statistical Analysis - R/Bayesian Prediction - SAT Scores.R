rm(list=ls())
data <- read.csv("/Users/ronitkathuria/Desktop/Study Abroad/Bayesian Data Analysis/Project/scores.csv")
data <- data[complete.cases(data[, c("Average.Score..SAT.Math.", "Average.Score..SAT.Reading.", "Average.Score..SAT.Writing.")]), ]
attach(data)
names(data)

data$Total.SAT.Score <- rowSums(data[,c("Average.Score..SAT.Math.", "Average.Score..SAT.Reading.", 
                                        "Average.Score..SAT.Writing.")], na.rm = TRUE)
total_sat_mean <- mean(data$Total.SAT.Score)
total_sat_sd <- sd(data$Total.SAT.Score)
total_sat_min <- min(data$Total.SAT.Score)
total_sat_max <- max(data$Total.SAT.Score)
total_sat_mean
total_sat_sd
total_sat_min
total_sat_max
hist(data$Total.SAT.Score, main="Distribution of Total SAT Scores", 
     xlab="Total SAT Score", col="blue", breaks=20)

n <- length(data$Total.SAT.Score)
mean_data <- mean(data$Total.SAT.Score)
var_data <- var(data$Total.SAT.Score)

m <- 0
c <- 0.01
a <- 0.01
b <- 0.01

m_ast <- (c * m + n * mean_data) / (c + n)
c_ast <- c + n
a_ast <- a + n
b_ast <- b + (n - 1) * var_data + (c * n * (m - mean_data)^2) / (c + n)

m_ast
c_ast
a_ast
b_ast

install.packages("nclbayes", repos="http://R-Forge.R-project.org")

library(nclbayes)

mu_range <- seq(mean_data - 3 * sd(data$Total.SAT.Score), 
                mean_data + 3 * sd(data$Total.SAT.Score), length.out = 1000)
tau_range <- seq(1 / (2 * var_data), 1 / (0.5 * var_data), length.out = 1000)

NGacontour(mu_range, tau_range, m, c, a, b, main = "Prior and Posterior Distributions")
NGacontour(mu_range, tau_range, m_ast, c_ast, a_ast, b_ast, add = TRUE, col = "red")
legend("topright", legend = c("Prior", "Posterior"), col = c("black", "red"), lty = 1)

M <- 10000
tau_post <- rgamma(M, shape = a_ast/2, rate = b_ast/2)
mu_post <- rnorm(M, mean = m_ast, sd = sqrt(1 / (c_ast * tau_post)))
plot(mu_post, tau_post, col = "darkgrey", main = 
       "Samples from Posterior Distribution", xlab = expression(mu), ylab = expression(tau))
NGacontour(mu_range, tau_range, m_ast, c_ast, a_ast, b_ast, add = TRUE, col = "red")

quantile(mu_post, c(0.025, 0.975))
quantile(tau_post, c(0.025, 0.975))

pred_mean <- m_ast
pred_scale <- (c_ast + 1) * b_ast / (c_ast * a_ast)
pred_df <- a_ast
hist(data$Total.SAT.Score, freq=FALSE, main="Histogram of Total SAT 
     Scores with Predictive Density", xlab="Total SAT Score", col="lightblue", breaks=20)
x_axis <- seq(min(data$Total.SAT.Score), max(data$Total.SAT.Score), by=1)
lines(x_axis, dt((x_axis - pred_mean) / sqrt(pred_scale), pred_df) / 
        sqrt(pred_scale), col="red", lwd=2)

plot(ecdf(data$Total.SAT.Score), main="Empirical and Predictive CDFs", 
     xlab="Total SAT Score", ylab="CDF")
lines(seq(min(data$Total.SAT.Score), max(data$Total.SAT.Score), by=1), 
      pt((seq(min(data$Total.SAT.Score), max(data$Total.SAT.Score), by=1) - pred_mean) 
         / sqrt(pred_scale), pred_df), 
      col="red")
legend("bottomright", legend=c("Empirical CDF", "Predictive CDF"), col=c("black", "red"), lty=1)

predictive_prob <- 1 - pt((1990 - pred_mean) / sqrt(pred_scale), pred_df)
predictive_prob

M <- 10000
pred_samples <- rnorm(M, mean = mu_post, sd = sqrt(1 / tau_post))
mc_predictive_prob <- mean(pred_samples > 1990)
mc_predictive_prob


library(MCMCpack)
mcmc_model <- MCMCregress(Total.SAT.Score ~ 1, data = data, 
                          b0 = 1500, B0 = 1/300^2, 
                          c0 = 0.001, d0 = 0.001)
summary(mcmc_model)
posterior_samples <- as.data.frame(mcmc_model)
set.seed(123)
pred_samples <- rnorm(10000, mean = posterior_samples$`(Intercept)`, sd = posterior_samples$S)
predictive_prob <- mean(pred_samples > 1990)
predictive_prob


















