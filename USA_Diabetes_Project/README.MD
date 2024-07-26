# Diabetes Health Indicators Dataset

This project is about using various machine learning models and deep learning Artificial Neural Networks (ANN) to predict whether a patient has diabetes or not using various health indicators. The data is from the USA **Centers for Disease Control and Prevention (CDC)**, collected annually as a survey to respondants. As the central aim is to predict diabetes in binary answers, classification algorithm was used through Decision Tree, Random Forest Ensemble, LightGBM, and XGBoost models with GridSesarch Hyperparameter tuning. To train ANN model, Tensorflow and Keras layer models were used with Keras Tuner for Hyperparameter tuning. This project was done with Python and Google Colaboratory environment. 

## Dataset

The project data was retrieved from Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv. The original dataset could be found from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv. This dataset was preprocessed beforehand so that the distribution of diabetes (0 or 1) would be equal, to prevent overfitting and training bias. Column information is contained inside the notebook at the beginning. 

## Data Preprocessing 

As the main objective is classificaiton and not regression, the preprocess stage of removing NULL values have been already done. Hence there wasn't much step needed for data preprocessing. 

## EDA (Exploratory Data Analysis)

For this project, EDA was heavily focused to analyze the relationship between each feature and how it actually affects diabetes. Heatmap has been used to investigate the correlation ratio between features for feature selection. After, selected features have been analyzed one-to-one with the diabetes_binary column with diabetes prevalance rate for accurate interpretation. Not only, as this project is a medical project, all data trends and valuable insights have been supported with actual medical study papers by various organizations. 

## Modeling

The models have been built using classification algorithm mainly with Decision Tree and Random Forest with LightGBM + XGBoost libraries used for performance purpose. For a further comparison, although the dataset was not massive to use neural networks, ANN model have been used for prediction as well. The models' hyperparamters were also tuned to a certain degree to produce the best performance for predictions. The training dataset and testing dataset was divided into 80 : 20 ratio, using stratified K-fold cross validation to increase performance and decreasing training bias by following the same diabetes_binary value distribution when training as well. After the training, the models were evaluated using scoring metrics mainly of accuracy, precision, recall, and overall f1-score. 

## Conclusion

After performance evaluation, the LightGBM model has been selected as the best performing model with the given data with the highest accuracy and f1-score. This is probably because LightGBM model shows a high performance when there are lots of data (data > 10000). Although LightGBM has the highest performance, the accuracy and f1-score wasn't able to achieve a high score in general which could be due to certain limitations such as limited hyperparameter tuning due to constrained GPU units or because of limited training data as the data was preprocessed beforehand to make it balanced by using under-sampling technique as the original data was 253,680 * 22 and the preprocessed 50:50 data that was used for this project is 70692 * 22. For further improvements, the original data on Kaggle could be used with more data, and if not, more in-depth hyperparameter tuning for these models or feature engineering methods could be a good start, or to use other advanced models if there are no improvements. The Artificial Neural Networks was not able to attain a great performance score compared to LightGBM and XGBoost, but better than Random Forest Ensemble algorithm, which might be the limitation of the model in terms of overfitting and bias. Further touch on the model with more conditions and manipulations could highly likely increase performance. 

## Sources

1. Antonopoulos, Andreas S., et al. "The Role of Inflammation in Atrial Fibrillation: A Comprehensive Review." *Journal of Atrial Fibrillation*, vol. 11, no. 1, 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5953551/.

2. American Heart Association. "Cholesterol Abnormalities & Diabetes." *American Heart Association*, https://www.heart.org/en/health-topics/diabetes/diabetes-complications-and-risks/cholesterol-abnormalities--diabetes.

3. National Center for Biotechnology Information. "Complications of Diabetes Mellitus." *NCBI Bookshelf*, https://www.ncbi.nlm.nih.gov/books/NBK592412/.

4. News-Medical. "Diabetes in Men versus Women." *News-Medical.net*, https://www.news-medical.net/health/Diabetes-in-Men-versus-Women.aspx.

5. Centers for Disease Control and Prevention. "Data & Research | Diabetes | CDC." *Centers for Disease Control and Prevention*, https://www.cdc.gov/diabetes/php/data-research/index.html#:~:text=Incidence%20among%20adults&text=Compared%20to%20adults%20aged%2018,and%20older%20(Table%203).

6. Cheng, Shao-Chun, et al. "Diabetes and Mortality in Pay-for-Performance Programs." *Health Policy*, 2023, https://www.sciencedirect.com/science/article/pii/S0168851023002026#:~:text=With%20higher%20education%2C%20diabetes%20is,mortality%20in%20diabetes%20P4P%20program.

7. Valdes-Ramos, Roxana, et al. "Nutrition and Inflammation in the Pathogenesis of Diabetic Retinopathy." *Journal of Immunology Research*, 2015, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4603875/.
