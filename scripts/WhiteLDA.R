# K CROSS FOLD VALIDATION FOR WHITE 
# recipes 
white_recipe <- recipe(quality ~ volatile.acidity + fixed.acidity + citric.acid 
                       + residual.sugar + chlorides + free.sulfur.dioxide 
                       +  total.sulfur.dioxide + density + pH + sulphates 
                       + alcohol, data = white_train) %>% step_dummy(all_nominal_predictors()) %>% step_normalize(all_predictors())


#lda model using cross validation
#set up model with mode classification and engine MASS
wlda_model <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS") 

#add model and recipe to the workflow
wlda_wkflow<- workflow() %>% 
  add_model(wlda_model) %>% 
  add_recipe(white_recipe)

#create a fit between the workflow and folded data
wlda_fit_cross <- fit_resamples(wlda_wkflow, white_fold)

# EVAL = TRUE
#determine the roc_auc of the LDA model on the folded training data
collect_metrics(wlda_fit_cross)
#_________________________________________

# FITTING THE MODEL TO LDA 

# LOAD IN DATA HERE USING load("WhiteLDA.R")
 
# fitting the model to training data
wlda_fit <- fit(wlda_wkflow, white_train)

# returning the accuracy, roc_auc, roc_auc curves, heatmap 
wlda_pred <- augment(wlda_fit, new_data = white_test) 
wlda_acc <- wlda_pred %>% accuracy(truth = quality, estimate = .pred_class)
wlda_rocauc <- wlda_pred %>% roc_auc(truth = quality, estimate = .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% mutate(model_type = "White LDA Model")
wlda_roccurve <- wlda_pred %>% roc_curve(quality, .pred_3,.pred_4, .pred_5 , .pred_6 , .pred_7, .pred_8) %>% autoplot()
wlda_confusionmatrix <- wlda_pred %>% conf_mat(truth = quality, estimate = .pred_class) %>% autoplot(type = "heatmap")

print(wlda_acc)
print(wlda_rocauc)
wlda_roccurve
wlda_confusionmatrix

# saving 
save(wlda_model, wlda_wkflow, wlda_fit ,wlda_fit_cross, wlda_pred, wlda_acc, wlda_rocauc, wlda_roccurve, wlda_confusionmatrix, file = "WhiteLDA.rda")


