##################### STAT 410 Final Project Code #########################
# reading in data
setwd("C:/Users/Kelly/Downloads/2021Sem2/STAT410/STAT410 Project")
getwd()

# data from kaggle: https://www.kaggle.com/datasets/equinxx/spotify-top-50-songs-in-2021
spotify = read.csv(file = 'data/spotify_top50_2021.csv', header = TRUE)
View(spotify)

# initial plots to observe relations between variables
plot(spotify$danceability, spotify$popularity)
plot(spotify$energy, spotify$popularity)
plot(spotify$key, spotify$popularity)
plot(spotify$loudness, spotify$popularity)
plot(spotify$mode, spotify$popularity)
plot(spotify$speechiness, spotify$popularity)
plot(spotify$acousticness, spotify$popularity)
plot(spotify$instrumentalness, spotify$popularity)
plot(spotify$liveness, spotify$popularity)
plot(spotify$valence, spotify$popularity)
plot(spotify$tempo, spotify$popularity)
plot(spotify$duration_ms, spotify$popularity)
plot(spotify$time_signature, spotify$popularity)

############ MLR Model ##############
# fitted model
fit = lm(popularity ~ danceability + energy + key + loudness + mode +
           speechiness + acousticness + instrumentalness + liveness + 
           valence + tempo + duration_ms + time_signature, data = spotify)
summary(fit)

# Pairwise plot
dat = spotify[,c(-1,-2,-3, -4)]
View(dat)
pairs(dat) # energy and loudness look correlated

# MLR Diagnostics
plot(fit) # looks a bit concerning

# seeing if log transformation fixes
fit_trans = lm(log(popularity) ~ danceability + energy + key + loudness + mode +
                 speechiness + acousticness + instrumentalness + liveness + 
                 valence + tempo + duration_ms + time_signature, data = spotify)
plot(fit_trans) # looks the same as before
# say that assumptions hold although there may be some concerns

# fitted values
plot(fitted(fit)); abline(h = c(0,1))

# fitted lines 
plot(fitted(fit), spotify$popularity); 
lines(fitted(fit), fitted(fit))

# other quantities 
coef(summary(fit)) 
(beta_hat_ols <- coef(fit)) # ols estimates
summary(fit)$sigma # s hat value
confint(fit, 
        level = 0.95) # 95% confidence intervals for the regression coefs

################## Ridge and Lasso ##################
# RIDGE
library(glmnet)
y = spotify$popularity
X = model.matrix(~ danceability*mode + energy + key + loudness +
                   speechiness + acousticness + instrumentalness + liveness + 
                   valence + tempo + duration_ms + time_signature, data = spotify)

# Now find an "optimal" lambda using cross-validation: 
fit.cv.r = cv.glmnet(X,  # Matrix of predictors (w/o intercept)
                     y,  # Response
                     alpha=0, # ridge 
                     lambda = seq(100, 0, by = -0.01) # lambda sequence
)
plot(fit.cv.r)

# Reasonable values for lambda:
fit.cv.r$lambda.min # Minimizes CV error^2
fit.cv.r$lambda.1se # largest lambda w/in 1 SE of lambda.min

beta.r.min <- coef(fit.cv.r, s = "lambda.min")[-2,]
beta.r.1se <- coef(fit.cv.r, s = "lambda.1se")[-2,]

cbind(beta_hat_ols, beta.r.min)
cbind(beta_hat_ols, beta.r.1se)
# the ridge estimates are different from the ols estimates

# LASSO
# Now find an "optimal" lambda using cross-validation: 
fit.cv.l = cv.glmnet(X,  # Matrix of predictors (w/o intercept)
                     y,  # Response
                     alpha=1, # lasso 
                     lambda = seq(100, 0, by = -0.01) # lambda sequence
)
plot(fit.cv.l)

# Reasonable values for lambda:
fit.cv.l$lambda.min # Minimizes CV error^2
fit.cv.l$lambda.1se # largest lambda w/in 1 SE of lambda.min

beta.l.min <- coef(fit.cv.l, s = "lambda.min")
beta.l.1se <- coef(fit.cv.l, s = "lambda.1se")

cbind(beta_hat_ols, beta.l.min)
cbind(beta_hat_ols, beta.l.1se)
# the lasso estimates are different from the ols estimates

############ picking a model using AIC and BIC ##############
library(leaps)
# considering using time signature as a factor instead of variable
#spotify$mode <- factor(spotify$mode)
#spotify$time_signature <- factor(spotify$time_signature)

# full model 
fit_full = lm(popularity ~ danceability + energy + key + loudness + mode +
                speechiness + acousticness + instrumentalness + liveness + 
                valence + tempo + duration_ms + time_signature, data = spotify)
# null model
fit_null <- lm(popularity ~ 1, data = spotify)

anova(lm(popularity ~ danceability, data = spotify), # Null model
      fit_full, # Full model
      test = 'F')

# AIC forward to determine which model with whichi predictors is the best
step(fit_null, list(upper=fit_full), direction='forward')
# now backwards
step(fit_null, list(upper=fit_full), direction='backward')
# picked the model lm(formula = popularity ~ danceability + time_signature, data = spotify)
# The algortihm stops at this model because the model after would have higher AIC values,
# so we know that model has the smallest AIC values
# Another explanation would be that removing another variable would not make the AIC value smaller,
# so the we do not need to remove further variables. We then have our best model with minimized AIC

# same thing but BIC
n <- nrow(spotify)
step(fit_null, list(upper=fit_full), direction='forward', k = log(n))
# now backwards
step(fit_null, list(upper=fit_full), direction='backward', k = log(n))
# picked the model lm(formula = popularity ~ danceability + time_signature, data = spotify)
# same as AIC

############ picking a model using Adj R Squared  ##############
# Adj R Squared
subsets <- regsubsets(popularity ~ danceability + energy + key + loudness + mode +
                        speechiness + acousticness + instrumentalness + liveness + 
                        valence + tempo + duration_ms + time_signature, data = dat, method='exhaustive', nvmax=14)
summary(subsets)
ar2 <- summary(subsets)$adjr2
plot(ar2)
abline(v=which(ar2==max(ar2)))
which(ar2==max(ar2)) # model of size 6 is the one we should select
# this model includes the variables danceability, energy, key, loudness, mode, time_signature

# VIF
library(car)
vif(fit_full) # none are over 5

############## looking into AIC and BIC model ###############
best_fit_ic <- lm(formula = popularity ~ danceability + time_signature, data = spotify)
summary(best_fit_ic)
AIC(best_fit_ic)
BIC(best_fit_ic)
summary(best_fit_ic)$adj.r.squared
plot(best_fit_ic)

# fitted lines 
plot(fitted(best_fit_ic), spotify$popularity); 
lines(fitted(best_fit_ic), fitted(best_fit_ic))

# Other quantities we care about:
coef(summary(best_fit_ic)) 
coef(best_fit_ic) # OLS estimates
summary(best_fit_ic)$sigma # S_hat
confint(best_fit_ic, 
        level = 0.95) # 95% confidence intervals for the regression coefs

############## looking into Adj R Squared model ###############
best_fit_ar2 <- lm(formula = popularity ~ danceability + energy + key + loudness + mode + time_signature, data = spotify)
summary(best_fit_ar2)
AIC(best_fit_ar2)
BIC(best_fit_ar2)
summary(best_fit_ar2)$adj.r.squared
plot(best_fit_ar2)

# fitted lines 
plot(fitted(best_fit_ar2), spotify$popularity); 
lines(fitted(best_fit_ar2), fitted(best_fit_ar2))

# Other quantities we care about:
coef(summary(best_fit_ar2)) 
coef(best_fit_ar2) # OLS estimates
summary(best_fit_ar2)$sigma # S_hat
confint(best_fit_ar2, 
        level = 0.95) # 95% confidence intervals for the regression coefs

################# non-linear models #####################
library(mgcv)
library(faraway)

# partially additive model:
add_fit = gam(popularity ~ danceability + energy + s(key) + loudness + mode +
                s(speechiness) + acousticness + s(instrumentalness) + liveness + 
                s(valence) + tempo + duration_ms + time_signature, data = spotify)
summary(add_fit)
plot(add_fit, shade = TRUE) # +/- 2*SE
# very large error bars for instrumentalness variable plot

AIC(add_fit) # smaller AIC
BIC(add_fit)
summary(add_fit)$r.sq
# fitted lines
plot(fitted(add_fit), spotify$popularity); 
lines(fitted(add_fit), fitted(add_fit))

########################### Interactions ###########################
# between a numerical and categorical
# pick variables i think would have some type of correlation 

fit_inter = lm(popularity ~ danceability*mode + energy + key + loudness +
                 speechiness + acousticness + instrumentalness + liveness + 
                 valence + tempo + duration_ms + time_signature, data = spotify)
summary(fit_inter)
AIC(fit_inter)
BIC(fit_inter)
summary(fit_inter)$adj.r.squared
# not a better model

plot(fitted(fit_inter), spotify$popularity); 
lines(fitted(fit_inter), fitted(fit_inter))

########################### Predictions ###########################
predict(add_fit, data.frame(danceability = .8, energy = .8, key = 1, loudness = -2,
                            mode = 1,speechiness = .03, acousticness = .01,
                            instrumentalness = 0, liveness = .04, valence = .8,
                            tempo = 170, duration_ms = 200000, time_signature = 4,),
        interval = 'prediction')

