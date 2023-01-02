# SVM 
library(tidyverse)
library(tidymodels)
library(ISLR)
library(kernlab)

svm_rbf_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("kernlab")

whiteRBF_fit <- svm_rbf_spec %>%
  set_args(cost = 10) %>%
  fit(quality ~ ., data = white_train)

augment(whiteSVM_fit, new_data = white_train) %>%
  conf_mat(truth = quality, estimate = .pred_class)
