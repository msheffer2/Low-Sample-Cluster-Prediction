# Set working directory
################################################################################
setwd("c:\\repos\\Low-Sample-Cluster-Prediction\\")

# Load libraries
################################################################################
library(dplyr)
#Explicityly called:  gplots

# Load Data
################################################################################
summary <- readr::read_csv("./data/summary.csv") %>%
  select(Data, Algorithm, starts_with("Test")) %>%
  filter(Data == "iv_dat") %>%
  select(-Data)
names(summary) <- c("Algorithm", "Accuracy", "Segment 1", "Segment 2", "Segment 3", 
                    "Segment 4", "Segment 5")

# Visualize the Data
################################################################################
plot_dat <- as.matrix(summary[2:7])
rownames(plot_dat) <- summary$Algorithm

gplots::heatmap.2(x=plot_dat,
                  cellnote=plot_dat,
                  notecol="black",
                  notecex=1.5,
                  dendrogram="none",
                  trace="none",
                  Colv="Accuracy",
                  col="redblue",
                  scale="column",
                  density.info="none",
                  key=FALSE,
                  main="Accuracy Metrics Summary",
                  lhei=c(.1, .9),
                  lwid=c(.1, .9))

rm(summary, plot_dat)







