red_recipe <- recipe(quality ~ volatile.acidity + fixed.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, data = red_train) %>% step_dummy(all_nominal_predictors()) %>% step_normalize(all_predictors())

# fitting the model
rlda_model <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS") 

# creating the workflow
rlda_wkflow <- workflow() %>% 
  add_model(rlda_model) %>% 
  add_recipe(red_recipe)

# fitting the model to training data
rlda_fit <- fit(rlda_wkflow, red_train)

# now using that fit to test the model on the testing data
rlda_res <- predict(rlda_fit, new_data = red_test %>%  select(-quality), type = "class" ) 
rlda_res <- bind_cols(rlda_res, red_test %>%  select(quality)) 

# returning our predictions and visualizations 
rlda_pred <- augment(rlda_fit, new_data = red_test)
rlda_acc <- rlda_pred %>% accuracy(truth = quality, estimate = .pred_class)
rlda_rocauc <- rlda_pred %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8)%>% mutate(model_type = "Red LDA Model")
rlda_roccurve <- rlda_pred %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
rlda_confusionmatrix <- rlda_pred %>% conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

# predictions, accuracy, roc auc, roc curve, confusion matrix 
print(rlda_acc)
print(rlda_res)
print(rlda_rocauc)
rlda_roccurve
rlda_confusionmatrix

save(rlda_model, rlda_wkflow, rlda_fit, rlda_pred, rlda_acc, rlda_rocauc, rlda_roccurve, rlda_confusionmatrix, file = "RedLDA.rda")
