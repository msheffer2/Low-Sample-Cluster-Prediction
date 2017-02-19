# Set working directory
################################################################################
setwd("c:\\repos\\Low-Sample-Cluster-Prediction\\")

# Load libraries
################################################################################
library(dplyr)
#Explicitly called = readr, caret

# Load Data
################################################################################
# NOTE: This data was cleaned and anonymized in steps taken prior to this codeup

temp <- readr::read_csv("./data/data.csv") %>%
  select(-id, -seg4) %>%
  rename(segment=seg5)

# Create train & scoring datasets
train <- filter(temp, is.na(segment)==FALSE)
score <- filter(temp, is.na(segment)==TRUE) %>%
  select(-segment)

save(score, file="./data/score.Rdata")
rm(temp, score)

# Create 5 Partitions of the Training Data for cross validation later
################################################################################
set.seed(3456)
partition <- caret::createFolds(train$segment, k=5, list = FALSE)
save(partition, file="./data/partition.Rdata")
rm(partition)

# Create different versions of the predictor datasets
#
# iv_dat = raw data (only zero variance removed)
# iv_sub = raw data, subset of uncorrelated variables
# iv_tdat = transformed (standardized) data (only zero variance removed)
# iv_tsbu = transformed (standardized) data, subset of uncorrelated variables
################################################################################
iv_dat <- select(train, -segment) %>%
  data.frame()

dv <- as.factor(train$segment)
save(dv, file="./data/dv.Rdata")

rm(train, dv)

#Remove variables with zero variance
zv <- caret::nearZeroVar(iv_dat, saveMetrics = TRUE)
zv <- zv[zv$zeroVar==TRUE,]
zv <- rownames(zv)
summary(iv_dat[zv])
iv_dat <- iv_dat[,!names(iv_dat) %in% zv]
rm(zv)

#Create Transform Dataset
#trans <- caret::preProcess(iv_dat, method = c("YeoJohnson"))
trans <- caret::preProcess(iv_dat, method = c("center", "scale"))
trans
iv_tdat <- predict(trans, iv_dat)
rm(trans)

#Create Datasets without highly correlated variables
corr <- cor(as.matrix(iv_dat))
highcorr <- caret::findCorrelation(corr, cutoff=.5)
length(highcorr)
names(iv_dat[highcorr])
iv_sub <- select(iv_dat, -(highcorr))
rm(highcorr, corr)

corr <- cor(as.matrix(iv_tdat))
highcorr <- caret::findCorrelation(corr, cutoff=.5)
length(highcorr)
names(iv_dat[highcorr])
iv_tsub <- select(iv_dat, -(highcorr))
rm(highcorr, corr)

#Save out iv datasets
save(iv_dat, file="./data/iv_dat.Rdata")
save(iv_sub, file="./data/iv_sub.Rdata")
save(iv_tdat, file="./data/iv_tdat.Rdata")
save(iv_tsub, file="./data/iv_tsub.Rdata")

rm(iv_dat, iv_sub, iv_tdat, iv_tsub)






