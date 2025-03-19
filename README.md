# Employee Performance Analysis
The training of the data has been done using the code from https://github.com/AkshayDusad/Employee-Performance-Analysis
PLease refer the above link if you face issues.

#  Project Summary

### Requirement           
   The following insights are expected from this project:
- Department wise performances.
- Top 3 Important Factors effecting employee performance.
- A trained model which can predict the employee performance based on factors as inputs.
- Streamlit app for visualizing the data and create an interesting UI 

### Analysis
- The data is supervised and categorical. The predictor variables are ordinal and a few among them are nominal. The target variable 'Performance Rating' is ordinal. 
- To analyze the data, various data processing techniques like Label Encoding and Standardization is used. Correlation Coeffecient is used to interpret the relationship between variables. The most important features selected are Department, Job Role, Environment Satisfaction, Last Salary Hike Percent, Work Life Balance, Experience Years At This Company, Experience Years In Current Role, Years Since Last Promotion, Years With Current Manager. 
- For training the data and predicting the target, algorithms used are Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes, K-Nearest Neighbor, XGBoost Classifier and Artificial Neural Network. 
- A separate analysis of Department wise Performance is carried out.

### Summary
   The project was done with the purpose of finding out factors which affected the Performance of the employees, training a model which accurately predicts the Performance Rating of the employee, analyzing the data to provide recommendations to improve the performance and gain insights from the analysis.
   The following steps were carried out:
1. Import the data provided, find out the predictor & target variables and look for missing values.
2. Analysis of Department wise performance as asked.
3. Label Encoding the ordinal columns.
4. Calculate correlation coeffecient to find out the relationship between variables and then select the important features for analysis.
5. Standardizing the data and splitting it into test and train.
6. Training the data using algorithms like Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes, K-Nearest Neighbor, XGBoost Classifier and Artificial Neural Network and checking the accuracy to find out which algorithm is the best.
7. Exporting the model with highest accuracy. 

### Results
-    Random Forest with GridSearchCV gives 93% accuracy. The features that are positively correlated are Environment Satisfaction, Last Salary Hike Percent & Worklife Balance. This means that if these factors increases, Performance Rating will increase. On the other hand, the features that are negatively correlated are Years Since Last Promotion, Experience Years at this Company, Experience years in Current Role & Years with Current Manager. This means that if these factors increases, Performance Rating will go down.
-    The top 3 features effecting employee performances are:
1. Employee Environment Satisfaction (39.5%)
2. Employee Last Salary Hike Percent (33.3%)
3. Years since last promotion (16.7%)

### Analysis and Insights
-    It was observed that the maximum accuracy was obtained when we used Random Forest with GridSearchCV which was 93%. XGBoost Classifier also yielded an accuracy of 92.8%. 
-    The important features that are positively correlated are Environment Satisfaction, Last Salary Hike Percent & Worklife Balance. This means that if these factors increases, Performance Rating will increase. On the other hand, the features that are negatively correlated are Years Since Last Promotion, Experience Years at this Company, Experience years in current role & Years with Current Manager. This means that if these factors increases, Performance Rating will go down.
-    We can conclude that the company should provide a better environment as it increases the performance drastically. The company should increase the salary of the employee from time to time and help them maintain a worklife balance. On the other hand, shuffling the manager from time to time will also affect performance. 

### Recommendation
- From the results, we can conclude that the company should provide a better environment as it increases the performance drastically. The company should increase the salary of the employee from time to time and help them maintain a worklife balance. On the other hand, shuffling the manager from time to time will also affect performance. 

### Data app using Streamlit
-  After identifying the algorithm with the best accuracy, this model is used to create an app using streamlit. SLiders and radio buttons are used to select the employee performance. Based on this selection, the performance rating of the employee will be updated.
-  Install  streamlit before running the app using the following command
##### pip install streamlit

### To run the app (run the following command from the root directory) 
##### streamlit run emp-perf-app.py
