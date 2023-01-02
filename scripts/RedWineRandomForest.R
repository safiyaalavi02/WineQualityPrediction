# Red Wine Random Forest 
knitr::opts_chunk$set(echo = TRUE)
library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
library(klaR) # for naive bayes
library(discrim)
library(glmnet)
library(rpart)
library(randomForest)
library(xgboost)
library(ranger)
library(vip)
library(lubridate)
library(dplyr)
library(ISLR)
library(rpart.plot)
library(janitor)
library(ggpubr)

tidymodels_prefer()


white <- read.csv("Wine Dataset/winequality-white.csv", sep = ";")

red <- read.csv("Wine Dataset/winequality-red.csv", sep = ";")

# Make Column Names Clean 
#colnames(white) <- c("Fixed_Acidity", "Volatile_Acidity","Citric_Acid", "Residual_Sugar", "Chlorides", "Free_Sulfur_Dioxide", "Total_Sulfur_Dioxide", "Density", "pH", "Sulfates", "Alcohol", "Quality")
#colnames(red) <- c("Fixed_Acidity", "Volatile_Acidity","Citric_Acid", "Residual_Sugar", "Chlorides", "Free_Sulfur_Dioxide", "Total_Sulfur_Dioxide", "Density", "pH", "Sulfates", "Alcohol", "Quality")
white %>% clean_names()
red %>% clean_names()

white <- white[white$quality < 9,]


# Change quality to factor
white$quality <- factor(white$quality, levels = c(3,4,5,6,7,8))
red$quality <- factor(red$quality, levels = c(3,4,5,6,7,8))

# Adding type
white$type <- "White"
red$type <- "Red"

# data frame of combined wine
combinedWine <- rbind(white, red)

set.seed(1234)
white_split <- white %>% 
  initial_split(prop = 0.7, strata = "quality")

white_train <- training(white_split)
white_test <- testing(white_split)

red_split <- red %>% 
  initial_split(prop = 0.7, strata = "quality")

red_train <- training(red_split)
red_test <- testing(red_split)

white_fold <- vfold_cv(white_train, v = 5)
red_fold <- vfold_cv(red_train, v = 5)

#_______________________________________________________________________________
# setting random forest model up
rrandfor <- rand_forest() %>% set_engine("ranger", importance = "impurity") %>% set_mode("classification")

rrandfor_wf <- workflow() %>%
  add_model(rrandfor %>%
              set_args(mtry = tune(), trees = tune(), min_n = tune())) %>% 
  add_formula(quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol)

# tuning the model to find the best arguments 
param_grid2 <- grid_regular(mtry(range = c(1,9)), trees(range = c(15,17)), min_n(range = c(30,50)), levels = 8)

rtune_res_randfor <- tune_grid(
  rrandfor_wf,
  resamples = red_fold,
  grid = param_grid2
)

rAutoPlotRF <- autoplot(rtune_res_randfor)

# collecting metrics to find best mean
rbest_rocauc1 <- collect_metrics(rtune_res_randfor) %>% arrange(desc(mean))
print(rbest_rocauc1)
rbest_metric1 <- select_best(rtune_res_randfor)

rrandfor_final <- rand_forest(mtry = 7, trees = 17, min_n = 32) %>% set_engine("ranger", importance = "impurity") %>% set_mode("classification")
rrandfor_fit_final <- fit(rrandfor_final, formula = quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol,data = red_train)

rVIP <- vip(rrandfor_fit_final)
rVIP

# extracting the metrics 
rrandfor_pred <- augment(rrandfor_fit_final, new_data = red_train) 
rrandfor_acc <- rrandfor_pred %>% accuracy(truth = quality, estimate = .pred_class) %>% mutate(model_type = "Red Random Forest Model")
rrandfor_rocauc <- rrandfor_pred %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Random Forest Model")
rrandfor_roccurve <- rrandfor_pred %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
rrandfor_confusionmatrix <- rrandfor_pred %>% conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(rrandfor_acc)
print(rrandfor_rocauc)
rrandfor_roccurve
rrandfor_confusionmatrix

# testing 
rrandfor_pred_test <- augment(rrandfor_fit_final, new_data = red_test) 
rrandfor_acc_test <- rrandfor_pred_test %>% accuracy(truth = quality, estimate = .pred_class) %>% mutate(model_type = "Red Random Forest Model")
rrandfor_rocauc_test <- rrandfor_pred_test %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Random Forest Model")
rrandfor_roccurve_test <- rrandfor_pred_test %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
rrandfor_confusionmatrix_test <- rrandfor_pred_test %>%
  conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(rrandfor_acc_test)
print(rrandfor_rocauc_test)
rrandfor_roccurve_test
rrandfor_confusionmatrix_test

# saving 
save(rrandfor, rrandfor_wf, param_grid2, rtune_res_randfor, rAutoPlotRF, rbest_rocauc1, rrandfor_final, rVIP, rrandfor_fit_final, rrandfor_acc,rrandfor_rocauc, 
     rrandfor_roccurve, rrandfor_confusionmatrix, 
     rrandfor_acc_test, rrandfor_rocauc_test,rrandfor_pred_test,rrandfor_acc_test,rrandfor_rocauc_test,
     rrandfor_roccurve_test, rrandfor_confusionmatrix_test, file = "RedWineRandomForest.rda")


