library(devtools)
install_github("modelplot/modelplotr")

#Business problesm: let's say that we work for this bank and our marketing colleagues have asked us to help to select the customers that 
#are most likely to respond to a term deposit offer. 
#For that purpose, we will develop a predictive model and create the plots to discuss the results with our marketing colleagues. 
#Since we want to show you how to build the plots, not how to build a perfect model, we'll use six of these columns in our example. 
#Here's a short description on the data we use:

#y: has the client subscribed a term deposit?
#duration: last contact duration, in seconds (numeric)
#campaign: number of contacts performed during this campaign and for this client
#pdays: number of days that passed by after the client was last contacted from a previous campaign
#previous: number of contacts performed before this campaign and for this client (numeric)
#euribor3m: euribor 3 month rate

#load the data and have a quick look at it:download bank data and prepare 
bank<- read.csv('bank-additional-full.csv',sep=";",stringsAsFactors=FALSE,header = T)

bank <- bank[,c('y','duration','campaign','pdays','previous','euribor3m')]

# rename target class value 'yes' for better interpretation
bank$y[bank$y=='yes'] <- 'term.deposit'

#explore data
str(bank)

# prepare data for training and train models 
test_size = 0.3
train_index =  sample(seq(1, nrow(bank)),size = (1 - test_size)*nrow(bank) ,replace = F)
train = bank[train_index,]
test = bank[-train_index,]

# estimate some models with caret...
# setting caret cross validation, here tuned for speed (not accuracy!)
fitControl <- caret::trainControl(method = "cv",number = 2,classProbs=TRUE)
# mnl model using glmnet package
mnl = caret::train(y ~.,data = train, method = "glmnet",trControl = fitControl)

# random forest using ranger package, here tuned for speed (not accuracy!)
rf = caret::train(y ~.,data = train, method = "ranger",trControl = fitControl,
                  tuneGrid = expand.grid(.mtry = 2,.splitrule = "gini",.min.node.size=10))



# ... and estimate some models with mlr

library('mlr')
task = mlr::makeClassifTask(data = train, target = "y")

# discriminant model 
lrn = mlr::makeLearner("classif.lda", predict.type = "prob")
lda = mlr::train(lrn, task)

#Install & load xgboost package
library('xgboost')

#xgboost model
lrn = mlr::makeLearner("classif.xgboost", predict.type = "prob")
xgb = mlr::train(lrn, task)

library(modelplotr)

# transform datasets and model objects into scored data and calculate deciles 
prepare_scores_and_deciles(datasets=list("train","test"),
                           dataset_labels = list("train data","test data"),
                           models = list("rf","mnl","xgb","lda"),
                           model_labels = list("random forest","multinomial logit","XGBoost","Discriminant"),
                           target_column="y")

# transform data generated with prepare_scores_and_deciles into aggregated data for chosen plotting scope 
plotting_scope(select_model_label = 'XGBoost',select_dataset_label = 'test data')

# plot the cumulative gains plot
plot_cumgains()

# plot the cumulative gains plot and annotate the plot at decile = 2
plot_cumgains(highlight_decile = 2)

# plot the cumulative lift plot and annotate the plot at decile = 2
plot_cumlift(highlight_decile = 2)

# plot the response plot and annotate the plot at decile = 1
plot_response(highlight_decile = 1)

# plot the cumulative response plot and annotate the plot at decile = 3
plot_cumresponse(highlight_decile = 3)

# plot all four evaluation plots and save to file
plot_all(save_fig = TRUE,save_fig_filename = 'Selection model Term Deposits')

# set plotting scope to model comparison
plotting_scope(scope = "compare_models")

# plot the cumulative response plot and annotate the plot at decile = 3
plot_cumresponse(highlight_decile = 3)
