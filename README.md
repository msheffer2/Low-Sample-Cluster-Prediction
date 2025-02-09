Low Sample Cluster Prediction for Database Scoring
--------------------------------------------------

Clustering data to uncover potential useful groupings is one of my
favorite analyses to perform. In my current job, I'm often are tasked
with uncovering market segments for pharmaceutical clients who are
targeting a subset of physicians for different promotional programs.
Primary research data is used to provide insights into treatment
attitudes but in almost every case, a database of prescribing behaviors
is also used to provide additional insights into the resulting clusters.
The prescribing data is also used to predict likely segment membership
for every physician that's of interest to the client, regardless of
their participation in the primary research.

### The Problem of Low Sample Size for Prediction

------------------------------------------------------------------------

Very often the number targeted physicians for promotional activities is
relatively small, perhaps only 20,000 targeted individuals. This is a
benefit since data for such a small universe can very easily be
manipulated on almost any computer. One particular problem, though, is
that physicians are costly when it comes to getting primary research
data, so sample sizes used for clustering tend to be very small, perhaps
around 300 individuals. This posses a big problem for the classification
task.

Best practice dictates the use of a training and testing data to assure
that a prediction isn't overfitted. The small sample size hampers my
ability to use a robust enough sample to train the data or to have a
sizable hold-out sample for post hoc testing; removing 30% of the
training data will likely hamper the model fit given how few cases there
are to represent the wide range of segments to predict but keeping the
test data set low limits the ability to protect against overfitting.

The solution, then is to eschew the use of typical 70/30 train/test
split and to instead use the entire dataset to train while using cross
validation to estimate accuracy and control for over-fitting. Many
predictive algorithms use cross validation for parameter tuning (Kuhn &
Johnson 2013 \[1\]) but cross validation has been shown to be just as
good as holdout data in some situations and is particularly useful when
it isn't possible to use separate testing data (Kohavi 1995 \[2\], Borra
& Di Ciaccio 2010 \[3\]). In this example, I illustrate how I created a
process to use cross validation when predicting cluster assignments in
low sample situations where a true test hold out was not possible.

### Notes for this Example

------------------------------------------------------------------------

The data for this example come from a real-world piece of research but
the data has been heavily modified and anomyzed so that it bears only
the vaguest of correlation to the original data since all aspects of the
research study and data must remain confidential. As a result, I cannot
provide background on the study or describe the variables used in the
analysis (all of the variable names have been obscured with a simple
number system).

Unfortunately, I cannot provide any information on the cluster analysis
without the ability to add this context so I only concentrate on the
classification task. Additionally, I do not illustrate any EDA or
variable importance information since this type of analysis didn't seem
as interesting without the context.

Last, the syntax files used here are an example of my process only; this
does not represent the real syntax files I used for the original study.
I used the code from the original study to develop an internal R package
called "segpred" to help me automate some of the less interesting parts
of the process and to assess many different types of predictive
algorithms in the shortest time possible. The package is not shown here
but I will show a summary of the results from the original study
(altered and anomyzed) to illustrate my process. In this example,
though, I provide syntax for a random forest predictive algorithm only.

### R syntax files I used to generate the material necessary for the repo:

------------------------------------------------------------------------

-   [01 -
    Codeup.R](https://github.com/msheffer2/Low-Sample-Cluster-Prediction/blob/master/01%20-%20Codeup.R)
    -- takes a precleaned and anomyzed dataset and manipulates it for
    the classification task to follow. It also shows the steps for data
    cleaning (removing zero variance variables, transforming variables,
    removing highly correlated variables) and preparation steps (setting
    up cross validation partitions) prior to classification.
-   [02 -
    Model.R](https://github.com/msheffer2/Low-Sample-Cluster-Prediction/blob/master/02%20-%20Model%20Code.R)
    -- lays out a few functions (also included in lazyfunctions.R) to
    predict cluster membership via the caret package with parameter
    tuning that makes use of parallel processing. This particular file
    only uses a random forest predictive algorithm but forms the basis
    of a function I wrote later to do the same types of tasks with
    different algorithms. It also includes functions to perform the
    cross validation and plot the accuracy and segment sensitivities for
    the cross-validation step.
-   [03 - Post Model
    Analytics.R](https://github.com/msheffer2/Low-Sample-Cluster-Prediction/blob/master/03%20-%20Model%20Comparison.R)
    -- Takes a summary file for multiple models and compares the
    predictive accuracy across them. It also shows a heatmap of the
    results to help identify a potential winner or to highlight
    potential problems.
-   [04 - Improved
    Model.R](https://github.com/msheffer2/Low-Sample-Cluster-Prediction/blob/master/04%20-%20Improved%20Model.R)
    -- an adaptation of 02 - Model.R that performs simplified
    predictions for two troublesome segments, adds the probability of
    membership as additional predictors to the original dataset, and
    then refits the model to much better effect. It also scores the fake
    "database" as is typical at the end of a process like this.
-   [lazyfunctions.R](https://github.com/msheffer2/Low-Sample-Cluster-Prediction/blob/master/lazyfunctions.R)
    -- this syntax file contains the functions originally written in
    02 - Model.R. This files is called in 04 - Improved Model.R rather
    than retyping the functions to keep the syntax files clean.

#### Technical Notes:

-   The work shown here is less "replicable" than some of the other
    repos because of the need to alter all the original data and to
    avoid showing lots of rather uninteresting syntax showing multiple
    modelling efforts. Instead, I provide pre-cleaned or pre-outputted
    data to streamline this process. The random forest predictions show
    here, though, should be completely reproducible.

### Analytical Highlights

------------------------------------------------------------------------

The original cluster work was part of a multi-step process to identify
marketable segments for the pharmaceutical client. Unfortunately, I
cannot describe the resulting segments but summaries of the segments
were well received by the client. They next step was to predict segment
membership based on available behavioral (prescribing) data only so that
their database of ~26,000 targets could be scored.

##### Table 1: Distribution of 5 Clusters in the Sample Data

<table style="width:44%;">
<colgroup>
<col width="13%" />
<col width="15%" />
<col width="15%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Segment</th>
<th align="center">Sample N</th>
<th align="center">Sample %</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1</td>
<td align="center">81</td>
<td align="center">21.6%</td>
</tr>
<tr class="even">
<td align="center">2</td>
<td align="center">91</td>
<td align="center">24.3%</td>
</tr>
<tr class="odd">
<td align="center">3</td>
<td align="center">72</td>
<td align="center">19.2%</td>
</tr>
<tr class="even">
<td align="center">4</td>
<td align="center">49</td>
<td align="center">13.1%</td>
</tr>
<tr class="odd">
<td align="center">5</td>
<td align="center">82</td>
<td align="center">21.9%</td>
</tr>
<tr class="even">
<td align="center">Total</td>
<td align="center">375</td>
<td align="center">100.0%</td>
</tr>
</tbody>
</table>

The resulting solution consisted of 5 clusters that were roughly equal
in size in the sample. Segment 4 was of particularly interest because it
tended to have the highest prescribers who also had the most positive
attitudes towards the treatment condition. The client was actually
surprised that this group was "so big" at even 13% of the sample given
how raw this type of treater is. As will be shown below, this likely
results from some sample error that draws in "better" respondents more
easily that "less valuable" respondents for the client.

#### Model Function for Random Forest

------------------------------------------------------------------------

Below is the function I wrote for this analysis (and which later formed
the basis of a package I wrote to make this easier for future cluster
predictions). I've included notes when necessary but I'd like to point
out just a few features I considered when I wrote this code: \* Uses
caret in this function for parameter tuning \* Takes into account
parallel processing, when possible, to assist in parameter tuning \*
Performs cross validation after training to assess the predictive
accuracy \* Collects the fit results and several other pieces of data
for plotting and archiving \* NOTE: I'm suppressing warnings for this
example only, but don't as a rule allow warnings to be suppressed in my
own production environment.

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

The example above predicted segment membership using a random forest but
in the real world, I would try many different types of predictive
algorithms. The training accuracy results below show why cross
validation is necessary.

##### Figure 2: Training Accuracy Output

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5
    ##          1 81  0  0  0  0
    ##          2  0 91  0  0  0
    ##          3  0  0 72  0  0
    ##          4  0  0  0 49  0
    ##          5  0  0  0  0 82
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9902, 1)
    ##     No Information Rate : 0.2427     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity             1.000   1.0000    1.000   1.0000   1.0000
    ## Specificity             1.000   1.0000    1.000   1.0000   1.0000
    ## Pos Pred Value          1.000   1.0000    1.000   1.0000   1.0000
    ## Neg Pred Value          1.000   1.0000    1.000   1.0000   1.0000
    ## Prevalence              0.216   0.2427    0.192   0.1307   0.2187
    ## Detection Rate          0.216   0.2427    0.192   0.1307   0.2187
    ## Detection Prevalence    0.216   0.2427    0.192   0.1307   0.2187
    ## Balanced Accuracy       1.000   1.0000    1.000   1.0000   1.0000

A perfect prediction! Too bad I don't believe it. It's surprisingly easy
to get really great training predictions like the one shown in the
confusion matrix above but this is exactly why we need to
cross-validate: a large drop in accuracy indicates over-fitting and the
CV results can provide a better picture of predictive accuracy.

##### Figure 3: Cross Validation Accuracy Output

![](README_files/figure-markdown_strict/unnamed-chunk-5-1.png)

The graph in Figure 3 summarizes the change in the accuracy of the
multi-class prediction in the training dataset (yellow bar), in each of
the 5 cross-validation folds (blue bars), and the average of the 5
cross-validation folds (red bar). The CV suggests that the model is
over-fit and that the accuracy is quite a bit below the perfect accuracy
indicated in the confusion matrix; the CV estimate of the accuracy for
this model is about 62%, not 100%.

##### Figure 4: Cross Validation Sensitivity Output

![](README_files/figure-markdown_strict/unnamed-chunk-6-1.png)

In some contexts, 62% might be an acceptable level of accuracy but I
prefer to achieve a higher level of accuracy for this type of problem.
What's more, I also prefer to have all segments (or at least the key
segments) be uniformly predicted in terms of accuracy. Figure 4 shows
the sensitivities for each segment and it shows how non-uniform the
accuracy is across the segments. The train sensitivity is shown in the
yellow boxes while the average of the CV sensitivities ins shown in the
red boxes. Segments 1, 2, and 5 are pretty accurately predicted
(&gt;77%) but Segments 3 and 4 are very poor in their accuracy (&lt;
20%). This is particularly problematic given how important Segment 4 was
the the client.

NOTE: In reality, EDA and variable importance might recommend some
changes to make (including data transformations). In my syntax files, I
actually perform this analysis on 4 different datasets to assess the
value of different combinations of variables and different data
transformations but this rarely improves the prediction over the raw
data. For this example, let's assume this the best model and that EDA or
variable importance (which I can't discuss) isn't helpful.

#### Model Comparison Summary

------------------------------------------------------------------------

Figure 5 (below) summarizes output from the segpred package in a
succinct table that describes the dataset, predictive algorithm, train,
and cv/test accuracy. In reality, this table would be much larger but
I've reduced it to just the test accuracies and only included the raw
dataset so that only the accuracy results are shown.

##### Figure 5: CV (Test) Accuracies for Various Predictive Models of Segment Membership

![](README_files/figure-markdown_strict/unnamed-chunk-7-1.png)

The heatmap in Figure 5 also sorts the results by the CV/Test Accuracy
(Acc.). In this example, Random Forest algorithms (both ranger and rf
are caret executions of a random forest algorithm) perform well overall
but the problem we saw about where Segments 3 and 4 are poorly predicted
plagues all of the algorithms. Also, while random forest does best
overall here, all of the algorithms do better than chance and could be
acceptable depending on the context.

#### Improving the Prediction for Segments 3 & 4

------------------------------------------------------------------------

As is often the case, I need to see if I could improve the predictive
accuracy for some of the segments (which subsequently would improve
accuracy overall). The syntax files attached illustrate how I
accomplished this but I ran to "pre-predictions" to see what each
person's probability was of being in Segment 3 or in Segment 4 and
included those two probabilities as additional predictors to use in the
model.

##### Figure 6: Cross Validation Accuracy Output

![](README_files/figure-markdown_strict/unnamed-chunk-8-1.png)

Figure 6 shows the improvement in accuracy with the additional
predictors (compared to Figure 3). Now the model retains it's accuracy
even after cross validations.

##### Figure 7: Cross Validation Sensitivity Output

![](README_files/figure-markdown_strict/unnamed-chunk-9-1.png)

Figure 7 also indicates that all 5 segments are uniformly well
predicted. OK, these are really almost too high to be believable and the
real accuracies/sensitivities in this research were nothing this high.
Nevertheless, this shows the general process and type of improvement or
result I expect even if I almost never expect to get anything this high!

##### Table 2: Distribution of 5 Clusters in the Sample Data & Scored Database

<table style="width:81%;">
<colgroup>
<col width="13%" />
<col width="15%" />
<col width="15%" />
<col width="18%" />
<col width="18%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Segment</th>
<th align="center">Sample N</th>
<th align="center">Sample %</th>
<th align="center">Database N</th>
<th align="center">Database %</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1</td>
<td align="center">81</td>
<td align="center">21.6%</td>
<td align="center">7,322</td>
<td align="center">27.3%</td>
</tr>
<tr class="even">
<td align="center">2</td>
<td align="center">91</td>
<td align="center">24.3%</td>
<td align="center">5,466</td>
<td align="center">20.4%</td>
</tr>
<tr class="odd">
<td align="center">3</td>
<td align="center">72</td>
<td align="center">19.2%</td>
<td align="center">2,310</td>
<td align="center">8.6%</td>
</tr>
<tr class="even">
<td align="center">4</td>
<td align="center">49</td>
<td align="center">13.1%</td>
<td align="center">759</td>
<td align="center">2.8%</td>
</tr>
<tr class="odd">
<td align="center">5</td>
<td align="center">82</td>
<td align="center">21.9%</td>
<td align="center">10,983</td>
<td align="center">40.9%</td>
</tr>
<tr class="even">
<td align="center">Total</td>
<td align="center">375</td>
<td align="center">100.0%</td>
<td align="center">26,840</td>
<td align="center">100.0%</td>
</tr>
</tbody>
</table>

The last task, then, is to score the database so that the client can use
the predictions to plan promotional or sales pieces targeted toward each
segment. The "positive" sample bias noted above is even more obvious now
once the estimated population sizes are shown in Table 2. Segment 4 was
a critical group and represented about 13% of the sample but in reality,
this group of individuals likely represents closer to 3% of the target
universe for the client (as an aside, this low percentage was not a
surprise to the client). Similarly, Segment 5 represent the lowest
opportunity for the client; it was only about 22% of the sample but
turned out to be a disappointing 41% of the target universe.

------------------------------------------------------------------------

##### References Cited

1.  Kuhh, Max. and Kjell Johnson. 2013. Applied Predictive Modelling.
    New York: Springer.
2.  Kohavi, Ron. 1995. "A Study of Cross-Validation and Bootstrap for
    Accuracy Estimation and Model Selection". Presented at the
    International joint Conference on Artificial Intelligence,
    Montreal, Quebec.
3.  Borra, Simone. and Agostino Di Ciaccio. 2010. "Measuring the
    Prediction Error: A Comparison of Cross-Validation, Bootstrap, and
    Covariance Penalty Methods". Computational Statistics and
    Data Analysis. 54: 2976-2989.
