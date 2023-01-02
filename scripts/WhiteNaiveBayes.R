# recipes 
white_recipe <- recipe(quality ~ volatile.acidity + fixed.acidity + citric.acid 
                       + residual.sugar + chlorides + free.sulfur.dioxide 
                       +  total.sulfur.dioxide + density + pH + sulphates 
                       + alcohol, data = white_train) %>% step_dummy(all_nominal_predictors()) %>% step_normalize(all_predictors())


#set up model with mode classification and engine kLaR
#we used set_args(use_kernel = FALSE) based on Lab 3
wnb_mod <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("klaR") %>% 
  set_args(usekernel = FALSE) 

#add model and recipe to the workflow
wnb_wkflow <- workflow() %>% 
  add_model(wnb_mod) %>% 
  add_recipe(white_recipe)

#create a fit between the workflow and folded data
wnb_fit_cross <- fit_resamples(wnb_wkflow, white_fold)

#determine the roc_auc of the Naive Bayes model on the folded training data
collect_metrics(wnb_fit_cross)