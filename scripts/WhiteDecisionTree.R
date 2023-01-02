# White Decision Tree 
# decision tree specification
wtree_spec <- decision_tree() %>%
  set_engine("rpart", model=TRUE)

# setting mode to classification
wtree_spec_class <- wtree_spec %>%
  set_mode("classification")

wclass_tree_fit <- wtree_spec_class %>%
  fit(quality ~ volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, data = white_train)


wDecisionTreePre <- wclass_tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(main="White Wine Decision Tree")

# augmented on training 
wDecisionTreeAccPre <- augment(wclass_tree_fit, new_data = white_train) %>%
  accuracy(truth = quality, estimate = .pred_class)

wDecisionTreeConfMatrixPre <- augment(wclass_tree_fit, new_data = white_train) %>%
  conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

# augmentsed on testing 
augment(wclass_tree_fit, new_data = white_test) %>%
  conf_mat(truth = quality, estimate = .pred_class)  %>% autoplot(type = "heatmap")

augment(wclass_tree_fit, new_data = white_test) %>%
  accuracy(truth = quality, estimate = .pred_class)

# tuning cost complexity 
wclass_tree_wf<- workflow() %>%
  add_model(wtree_spec_class %>% 
              set_args(cost_complexity = tune())) %>% 
  add_formula(quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol)

param_grid <- grid_regular(cost_complexity(range = c(-3,-1)), levels = 10)

tune_res_white <- tune_grid(
  wclass_tree_wf,
  resamples = white_fold,
  grid = param_grid,
  metric = metric_set(accuracy)
)

wAutoPlot <- autoplot(tune_res_white)

# extracting the best cost complexitiy parameter
wbest_rocauc <- collect_metrics(tune_res_white) %>% arrange(desc(mean))

wbest_complexity <- select_best(tune_res_white)

wclass_tree_final <- finalize_workflow(wclass_tree_wf, wbest_complexity)

wclass_tree_final_fit <- fit(wclass_tree_final, data = white_train)

wDecisionTree <- wclass_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot(main="White Wine Decision Tree")

# augmented on training 
wdectree_pred <- augment(wclass_tree_final_fit, new_data = white_train) 
wdectree_acc <- wdectree_pred %>% accuracy(truth = quality, estimate = .pred_class)
wdectree_rocauc <- wdectree_pred %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Decision Tree Model")
wdectree_roccurve <- wdectree_pred %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
wdectree_confusionmatrix <- augment(wclass_tree_final_fit, new_data = white_train) %>%
  conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(wdectree_acc)
print(wdectree_rocauc)
wdectree_roccurve
wdectree_confusionmatrix

# augmented on testing 
wdectree_pred_test <- augment(wclass_tree_final_fit, new_data = white_test) 
wdectree_rocauc_test <- wdectree_pred_test %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Decision Tree Model")
wdectree_roccurve_test <- wdectree_pred_test %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
wDecisionTreeAccTest <- augment(wclass_tree_final_fit, new_data = white_test) %>%
  conf_mat(truth = quality, estimate = .pred_class) 
wDecisionTreeConfMatrixTest <-augment(wclass_tree_final_fit, new_data = white_test) %>%
  accuracy(truth = quality, estimate = .pred_class)

# saving files 
save(wtree_spec, wtree_spec_class, wclass_tree_fit, wAutoPlot, wDecisionTreePre, wDecisionTreeAccPre, wDecisionTreeConfMatrixPre, wAutoPlot, wbest_rocauc, wDecisionTree, wAutoPlot, 
     wdectree_pred,wdectree_acc, wdectree_rocauc, wdectree_roccurve, wdectree_confusionmatrix, wdectree_pred_test, wdectree_rocauc_test,wdectree_roccurve_test,
     wclass_tree_final_fit, wDecisionTreeAccTest, wDecisionTreeConfMatrixTest, file = "WhiteWineDecisionTree.rda")
