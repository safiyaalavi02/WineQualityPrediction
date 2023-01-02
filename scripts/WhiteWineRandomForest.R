# White Wine Random Forest 

# setting random forest model up
wrandfor <- rand_forest() %>% set_engine("ranger", importance = "impurity") %>% set_mode("classification")

wrandfor_wf <- workflow() %>%
  add_model(wrandfor %>%
              set_args(mtry = tune(), trees = tune(), min_n = tune())) %>% 
  add_formula(quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol)

# tuning the model to find the best arguments 
param_grid1 <- grid_regular(mtry(range = c(1,9)), trees(range = c(15,17)), min_n(range = c(30,50)), levels = 8)

wtune_res_randfor <- tune_grid(
  wrandfor_wf,
  resamples = white_fold,
  grid = param_grid1,
  metric = metric_set(accuracy)
)

wAutoPlotRF <- autoplot(wtune_res_randfor)


# collecting metrics to find best mean
wbest_rocauc1 <-collect_metrics(wtune_res_randfor) %>% arrange(desc(mean))

wbest_metric1 <- select_best(wtune_res_randfor)

# wrandfor_final <- finalize_workflow(wrandfor_wf, wbest_metric1)
# wrandfor_fit_final <- fit(wrandfor_final, data = white_train)

wrandfor_final <- rand_forest(mtry = 2, trees = 17, min_n = 30) %>% set_engine("ranger", importance = "impurity") %>% set_mode("classification")
wrandfor_fit_final <- fit(wrandfor_final, formula = quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, data = white_train)

wVIP <- vip(wrandfor_fit_final)
wVIP 

# extracting the metrics 
wrandfor_pred <- augment(wrandfor_fit_final, new_data = white_train) 
wrandfor_acc <- wrandfor_pred %>% accuracy(truth = quality, estimate = .pred_class) %>% mutate(model_type = "White Random Forest Model")
wrandfor_rocauc <- wrandfor_pred %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Random Forest Model")
wrandfor_roccurve <- wrandfor_pred %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
wrandfor_confusionmatrix <- augment(wrandfor_fit_final, new_data = white_train) %>%
  conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(wrandfor_acc)
print(wrandfor_rocauc)
wrandfor_roccurve
wrandfor_confusionmatrix

# testing 
wrandfor_pred_test <- augment(wrandfor_fit_final, new_data = white_test) 
wrandfor_acc_test <- wrandfor_pred_test %>% accuracy(truth = quality, estimate = .pred_class) %>% mutate(model_type = "White Random Forest Model")
wrandfor_rocauc_test <- wrandfor_pred_test %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White Random Forest Model")
wrandfor_roccurve_test <- wrandfor_pred_test %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
wrandfor_confusionmatrix_test <- augment(wrandfor_fit_final, new_data = white_test) %>%
  conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(wrandfor_acc_test)
print(wrandfor_rocauc_test)
wrandfor_roccurve_test
wrandfor_confusionmatrix_test

# saving 
save(wrandfor, wrandfor_wf, param_grid1, wtune_res_randfor, wAutoPlotRF, wbest_rocauc1, wrandfor_final, wVIP, wrandfor_fit_final, wrandfor_pred, 
     wrandfor_acc, wrandfor_rocauc, wrandfor_roccurve, wrandfor_confusionmatrix, 
     wrandfor_pred_test, wrandfor_acc_test, wrandfor_rocauc_test,  wrandfor_roccurve_test, wrandfor_confusionmatrix_test,
     file = "WhiteWineRandomForest.rda")

