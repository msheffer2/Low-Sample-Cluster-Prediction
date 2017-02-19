# Set working directory
################################################################################
setwd("c:\\repos\\Low-Sample-Cluster-Prediction\\")

# Load libraries
################################################################################
library(caret)
library(randomForest)
library(doParallel)
library(dplyr)
#explicityly called:  descr, scales

#Instead of typing functions from original model code, I've saved them out as
#lazyfunctions.R, so loading them here.  
#Generally, anything I source is acandidate for a function in the future and
# in this case, the lazyfunctions code was used to build segpred function.
source("lazyfunctions.R")

# Load Data
################################################################################
load("./data/iv_dat.Rdata")
load("./data/partition.Rdata")
load("./data/dv.Rdata")

# Create New Predictor - Probability of Being in Segment 3
################################################################################
seg3 <- ifelse(dv==3, 1, 2) %>%
  as.factor()
pred3 <- lazypred(iv_dat, seg3, partition)
save(pred3, file="./output/pred3.Rdata")

prob3 <- predict(pred3$fit, iv_dat, type="prob")[1]
names(prob3) <- "prob3"
rm(seg3)

# Create New Predictor - Probability of Being in Segment 4
################################################################################
seg4 <- ifelse(dv==4, 1, 2) %>%
  as.factor()
pred4 <- lazypred(iv_dat, seg4, partition)
save(pred4, file="./output/pred4.Rdata")

prob4 <- predict(pred4$fit, iv_dat, type="prob")[1]
names(prob4) <- "prob4"
rm(seg4)

# Merge new predictors and remodel
################################################################################
iv_dat.2 <- bind_cols(iv_dat, prob3, prob4)

set.seed(3456)
mod1.2 <- lazypred(iv_dat.2, dv, partition)
save(mod1.2, file="./output/mod1.2.Rdata")

#Plotting Predictive Accuracy for Train & CV Test
acc_plot(mod1.2$a_dat)

#Plotting Segment Sensitivites for Train & CV Test
sens_plot(mod1.2$s_dat)

rm(iv_dat, iv_dat.2, prob3, prob4, partition, acc_plot, sens_plot, lazypred)

# Score the database
################################################################################
load("./data/score.Rdata")

#Add New predictors for Segments 3 and 4
prob3 <- predict(pred3$fit, score, type="prob")[1]
names(prob3) <- "prob3"

prob4 <- predict(pred4$fit, score, type="prob")[1]
names(prob4) <- "prob4"

score <- bind_cols(score, prob3, prob4)
rm(prob3, prob4, pred3, pred4)

#Predict Segment Membership
scored <- predict(mod1.2$fit, score, type="raw")

results <- cbind(descr::freq(dv, plot=FALSE)[,1],
                 scales::percent(round(descr::freq(dv, plot=FALSE)[,2]/100, digits=3)),
                 scales::comma(descr::freq(scored, plot=FALSE)[,1]),
                 scales::percent(round(descr::freq(scored, plot=FALSE)[,2]/100, digits=3)))

results <- data.frame(results) %>%
  mutate(Segment = row.names(results)) %>%
  select(Segment, everything())
names(results) <- c("Segment", "Sample N", "Sample %", "Database N", "Database %")
save(results, file="./output/results.Rdata")




