# Netflix Stock Price Prediction Project

This project is a time series forecasting project using LSTM and GRU recurrent neural networks to predic Netflix's stock close price. The central aim of this project is to use models to predict the stock price hence Recurrent Neural Networks of Long Short-Term Memory network and Gated Recurrent Unit network was built as these networks are very accurate and suitable for time series by their recurrent mechanism. To train the RNN models, Tensorflow and Keras layer models were used with Keras Tuner for Hyperparameter tuning. This project was done with Python and Google Colaboratory environment. 

## Dataset

The project data was retrieved from Kaggle: https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction. The dataset consists 7 columns: **Date** (Everyday Price), **Open** (The price of the stock when opened), **Close** (The price of the stock when closed), **High** (The price of the stock when high), **Low** (The price of the stock when low), **Adj Close** (The adjusted close price), and **Volume** (The amount of stock trades volume). The date is from 2018-02-05 to 2022-02-04, comprised of 4 years data. The stock price is in US dollar currency.

## Data Preprocessing 

The data was preprocessed by converting the date column type which was originally in string into datetime. There were no missing values, but there were outliers for the volume column. As the outliers could affect the model's performance, the whole data was scaled so that it is between 0 and 1 using MinMax scaling method before training the model. 

## EDA (Exploratory Data Analysis)

The EDA was conducted through generally visualizing the stock price and the volume at first to see the overall graph trend of each data column. The main focus for this project was to predict the closed price of the stock, hence more in-depth EDA was applied to Close Price data by plotting regression line to analyze periodicity and patterns in yearly interval from 2018-2019, 2019-2020, 2020-2021, and 2021-2022. After observing patterns, the moving average price was also visualized by 5 days, 10 days, 30 days, and 60 days. 

## Modeling

To train the model, after applying data scaling, the dataset was prepared by creating time series sequences. This is the most important part of the modeling process as it is very crucial for LSTM and GRU recurrent neural networks to have time series sequences intervals to train and capture long-term dependencies. Hence the time series sequences were built using sequence length of 3 hence 3 consecutive days for each sequence. Then the first basic LSTM model was created using two LSTM layers and two dense layers with dropout layers in between to minimize overfitting. The model was trained using ModelCheckPoint callbacks to save the best trained model for prediction. After the model was trained, the data was scaled back to original scale for evaluation, and the model was evaluated using two metrics: RMSE (Root Mean Squared Error) and R2-Score. The LSTM model had to be improved, so Hyperparameter Tuning using Bayesian Optimization was applied. For the tuning process, recurrent dropout rate was also included with objective tuning. After tuning, the model had a satisfying result of RMSE of 21 and R2 score of 0.91. Lastly, the GRU model was created and had the best performance even without tuning. The base GRU model was created using two GRU layers and two dense layers with dropout layers in between as well to prevent overfitting. It was trained with the same ModelCheckpoint callback to save the best trained model. The GRU model performanced robustly well by having a RMSE of 14.8 and with a R2 score of 0.96. 

## Conclusion

After creating three models which were basic LSTM model (without tuning), LSTM model with optimization (Tuning), and GRU model (without optimization), the GRU model outperformed the rest of the two models with a very high R2 score and low RMSE. Although there are some limitations of how the LSTM model was trained with the number of neurons for each layer, training rates, number of epochs and trials, and etc, still the GRU model had a significantly high scores with good performance in predictions. Hence for time series forecastig, when comparing LSTM and GRU, GRU network could be more of a suitable choice to use. If more in-depth tuning is applied to LSTM layer, the LSTM layer is also expected to have a equal satisfactory performance as the GRU model. 
