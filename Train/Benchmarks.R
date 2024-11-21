##################################
# Benchmarks for load prediction #
# 9th of August 2023             #
##################################

library(fpp2)

data <- read.csv("Compiled.csv")
load <- c(data$load)


# 1. Naive forecast

forecast_1 = naive(load)
mae_1 = mean(abs(forecast_1$residuals), na.rm=T)
# MAE = 25655.309


# 2. Seasonal naive forecast

# Daily
forecast_2_daily = snaive(ts(load,freq=24))
mae_2_daily = mean(abs(forecast_2_daily$residuals), na.rm=T)
# MAE = 53567.9

# Weekly
forecast_2_weekly = snaive(ts(load,freq=168))
mae_2_weekly = mean(abs(forecast_2_weekly$residuals), na.rm=T)
# MAE = 36635.161

# Yearly
forecast_2_yearly = snaive(ts(load,freq=8760))
mae_2_yearly = mean(abs(forecast_2_yearly$residuals), na.rm=T)
# MAE = 100568.695


# 3. Random walk with drift forecast

forecast_3 = c(forecast_1$fitted + lm(load ~ X, data=data)$coefficients[[2]])
mae_3 = mean(abs(forecast_3 - load), na.rm=T)
# MAE = 25654.822


# 4. Multiple linear regression

drop <- c("X","date","day","year")
data = data[,!(names(data) %in% drop)]

fit = lm(load~.,data)
mae_lm = mean(abs(fit$residuals), na.rm=T)
# MAE = 55780.038

summary(fit)

data$prev <- c(NA, data$load[-length(data)])
fit2 = lm(load~.,data)
mae_lm2 = mean(abs(fit2$residuals), na.rm=T)
# MAE = 60.432 !!!

summary(fit2)
