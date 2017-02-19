# Set working directory
################################################################################
setwd("c:\\repos\\Low-Sample-Cluster-Prediction\\")

# Load libraries
################################################################################
library(caret)
library(randomForest)
library(doParallel)
library(dplyr)
#Explicityly called:  reshape2, scales

# Load Data
################################################################################
load("./data/dv.Rdata")
load("./data/partition.Rdata")
load("./data/iv_dat.Rdata")
load("./data/iv_sub.Rdata")
load("./data/iv_tdat.Rdata")
load("./data/iv_tsub.Rdata")

# Model Function
################################################################################
# NOTE:  this code served as the basis of a function I wrote to handle
# multiple types of predictive models for segmentation classification
# called segpred that can do this easily for 11 different predictive algorithms.
# For this example, I'm just using the Random Forest model but this code served
# as the basis for the function.

lazypred <- function(tdata, class, partition){
  #caret settings
  fitControl <- trainControl(method = "CV", 
                             number = 5, 
                             allowParallel = TRUE, 
                             summaryFunction=multiClassSummary)
  cl <- makeCluster(10)
  registerDoParallel(cl)
  
  #Fit the model
  set.seed(3456)
  
  #NOTE: ONLY SUPRESSING WARNINGS FOR THIS EXAMPLE!
  options(warn=-1)
  #      In this case, CV results in some missing resample values in the 
  #      performance measures inside caret, which generates a warning.  
  #      To keep the output clean, I'm suppressing all warnings

  fit <- train(tdata, 
               class,
               method = "rf",
               trControl = fitControl,
               metric = "Accuracy",
               nthread = 10)

  #Predict the model
  pred <- predict(fit, tdata, type="raw")
  
  #Train Accuracy
  acc0 <- confusionMatrix(pred, class)
  
  #Cross-validate the model
  for (i in 1:5){
    set.seed(3456)
    f <- train(tdata[partition != i,], 
               class[partition != i],
               method = "rf",
               trControl = fitControl,
               metric = "Accuracy",
               nthread = 10)
    p <- predict(f, tdata[partition == i,], type="raw")
    a <- confusionMatrix(p, class[partition == i])
    assign(paste0("acc", i), a)
    rm(f, p, a)
  }  
  #NOTE:  Returning warnings to normal
  options(warn=0)
  
  stopCluster(cl)
  
  #Compile the confusion matrices
  acc_dat <- list(acc0, acc1, acc2, acc3, acc4, acc5)
  
  #Pull out sensitivity data for plotting later
  s_dat <- rbind(as.matrix(acc0, what="classes")[1,], 
                 as.matrix(acc1, what="classes")[1,],
                 as.matrix(acc2, what="classes")[1,],
                 as.matrix(acc3, what="classes")[1,],
                 as.matrix(acc4, what="classes")[1,],
                 as.matrix(acc5, what="classes")[1,])
  
  #Pull out the accuracy data for plotting later
  a_dat <- rbind(round(acc0$overall['Accuracy'], digits=3),
                 round(acc1$overall['Accuracy'], digits=3),
                 round(acc2$overall['Accuracy'], digits=3),
                 round(acc3$overall['Accuracy'], digits=3),
                 round(acc4$overall['Accuracy'], digits=3),
                 round(acc5$overall['Accuracy'], digits=3))
  
  a_train <- scales::percent(a_dat[1])
  a_test <- scales::percent(mean(a_dat[2:6]))
  a_summary <- sprintf("Train Accuracy = %s, CV Accuracy = %s", a_train, a_test)
  print(a_summary)
  
  #Compile output
  output <- list(s_dat=s_dat, a_dat=a_dat,  acc_dat=acc_dat, fit=fit) 
  return(output)
}

acc_plot <- function(a_dat){
  #Data Prep
  a_dat[7] <- mean(a_dat[2:6])
  a_dat <- data.frame(a_dat)
  names(a_dat) <- c("acc")
  a_dat$num <- c(1:7)
  a_dat$lab <- c("Train Data", "CV Fold 1", "CV Fold 2", "CV Fold 3", "CV Fold 4",
                 "CV Fold 5", "Average CV")
  a_dat$pos <- a_dat$acc + .03
  
  #Plot
  colors <- c("darkgoldenrod1", "cadetblue1", "cadetblue2", "cadetblue3", 
              "cadetblue", "cadetblue4", "red")
  
  a_plot <- ggplot(a_dat, aes(x=reorder(lab, num), y=acc)) + 
    geom_bar(position=position_dodge(), stat="identity", colour=colors, fill=colors) +
    labs(x="", title="Train & CV Predictive Accuracy", y="Accuracy") + 
    scale_y_continuous(labels=scales::percent, limits=c(0,1.1), breaks=c(0,.5,1)) + 
    geom_text(aes(label = sprintf("%1.1f%%", 100*acc), y = pos)) + 
    theme(text = element_text(size=15), 
          panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          axis.ticks=element_blank(),
          legend.position="none")
  print(a_plot)
}

sens_plot <- function(s_dat){
  #Data Prep
  s_dat <- rbind(s_dat, apply(s_dat[2:6,],2, mean))
  s_dat <- round(s_dat, digits=3) * 100
  s_dat <- reshape2::melt(s_dat)
  names(s_dat) <- c("num", "seg", "sensitivity")
  s_dat$seg <- as.factor(paste0("Segment ", s_dat$seg))
  s_dat$num <- factor(s_dat$num, levels = c(1:7), 
                      labels = c("Train Data", "CV Fold 1", "CV Fold 2", 
                                 "CV Fold 3", "CV Fold 4", "CV Fold 5", 
                                 "Average CV"))
  s_dat$lab1 <- ifelse(s_dat$num == "Average CV", s_dat$sensitivity, "")
  s_dat$lab2 <- ifelse(s_dat$num == "Train Data", s_dat$sensitivity, "")
  
  #Plot
  colors <- c("darkgoldenrod1", "cadetblue1", "cadetblue2", "cadetblue3", 
              "cadetblue", "cadetblue4", "red")
  
  s_plot <- ggplot(s_dat, aes(y=sensitivity, x=seg, color=num, shape=num)) +
    geom_point(size=6) +
    scale_shape_manual(name="", values=c(15, 1, 1, 1, 1, 1, 15)) + 
    scale_y_continuous(limits=c(0,100), breaks=c(0,20,40,60,80,100)) +
    theme_bw() +
    theme(panel.grid.major.y = element_blank()) + 
    labs(x="", title="Segment Sensitivities in Train & CV", y="Sensitivity") + 
    scale_color_manual(name="", values=colors) + 
    geom_text(aes(label=lab1), hjust=-.5, show.legend=FALSE) +
    geom_text(aes(label=lab2), hjust=-.5, show.legend=FALSE)
  print(s_plot)
}

# Mode #1:  iv_dat
################################################################################
mod1 <- lazypred(iv_dat, dv, partition)
save(mod1, file="./output/mod1.Rdata")

#Look at training accuracy
mod1$acc_dat[[1]]

#Look at fit summary
mod1$fit

#Plotting Predictive Accuracy for Train & CV Test
acc_plot(mod1$a_dat)

#Plotting Segment Sensitivites for Train & CV Test
sens_plot(mod1$s_dat)

# Mode #2 (iv_sub), #3 (iv_tdat), #4 (iv_tsub)
################################################################################
mod2 <- lazypred(iv_sub, dv, partition)
mod3 <- lazypred(iv_tdat, dv, partition)
mod4 <- lazypred(iv_tsub, dv, partition)










