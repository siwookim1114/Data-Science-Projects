# Diabetes Health Indicators Dataset

This project is about using various machine learning models to predict whether a patient has diabetes or not using various health indicators. The data is from the USA **Centers for Disease Control and Prevention (CDC)**, collected annually as a survey to respondants. As the central aim is to predict diabetes in binary answers, classification algorithm was used through Decision Tree, Random Forest Ensemble, LightGBM, and XGBoost models with GridSesarch Hyperparameter tuning. This project was done with Python and Google Colaboratory environment. 

** Dataset

The project data was first retrieved from Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset. However for training purposes, the dataset was preprocessed beforehand so that the distribution of diabetes (0 or 1) would be equal, to prevent overfitting and training bias. Column information is contained inside the notebook at the beginning. 

** Data Preprocessing 

As the main objective is classificaiton and not regression, the preprocess stage of removing NULL values have been already done. Hence there wasn't much step needed for data preprocessing. 

** EDA (Exploratory Data Analysis)

For this project, EDA was heavily focused to analyze the relationship between each feature and how it actually affects diabetes. Heatmap has been used to investigate the correlation ratio between features for feature selection. After, selected features have been analyzed one-to-one with the diabetes_binary column with diabetes prevalance rate for accurate interpretation. Not only, as this project is a medical project, all data trends and valuable insights have been supported with actual medical study papers by various organizations. 

** Modeling

The models have been built using classification algorithm mainly with Decision Tree and Random Forest with LightGBM + XGBoost libraries used for performance purpose. The models' hyperparamters were also tuned to a certain degree to produce the best performance for predictions. The training dataset and testing dataset was divided into 80 : 20 ratio, using stratified K-fold cross validation to increase performance and decreasing training bias by following the same diabetes_binary value distribution when training as well. After the training, the models were evaluation using scoring metrics mainly of accuracy, precision, recall, and overall f1-score. 

** Conclusion

After performance evaluation, the LightGBM model has been selected as the best performing model with the given data with the highest accuracy and f1-score. This is probably because LightGBM model shows a high performance when there are lots of data (data > 10000). Although LightGBM has the highest performance, the accuracy and f1-score wasn't able to achieve a high score in general which could be due to certain limitations such as limited hyperparameter tuning due to constrained GPU units or because of limited training data as the data was preprocessed beforehand to make it balanced by using under-sampling technique as the original data was 253,680 * 22 and the preprocessed 50:50 data that was used for this project is 70692 * 22. For further improvements, the original data on Kaggle could be used with more data, and if not, more in-depth hyperparameter tuning for these models could be a good start, or to use other advanced models if there are no improvements. 
