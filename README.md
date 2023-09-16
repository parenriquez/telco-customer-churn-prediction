#### **Telco Customer Churn Prediction with Logistic Regression**

#### **Abstract**

**Churn** is the movement of individuals OUT of a collective group over a specific period.

It is a good indicator of growth potential. It tracks lost customers, while growth tracks new customers. If churn is higher than growth, your business is getting smaller.

***Predicting churn*** is a good way to create proactive marketing campaigns targeted at the customers that are about to churn.
If successful, it saves the company from unnecessary expenses, and protects revenue from depletion.

During churn prediction:

1. We identify *at-risk* customers
2. We identify strategies to lower churn and increase customer retention

It provides a clear picture on the state and quality of the business, shows customer satisfaction with their product or service, and allows for comparison with competitors to gauge customer turnover at the same timeframe.

======================================================================================================================================================

In this repository, we will try to learn about Logistic Regression, using a dataset from the [Telco Customer Churn (11.1.3+)](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) data module is in the *Base Samples* provided by IBM Cognos Analytics.

It contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3.

The module contains 5 tables:

- Demographics
- Location
- Population
- Services
- Status

The tables were matched, cleaned, and combined into a singular dataset with 31 features and 1 target variable. **21** of the 31 features are *categorical*, while **10** are *numerical*. The target variable is categorical, written as either 1 or 0.

The final dataset is imbalanced – 73% are not churning (0), while only 27% are (1). You can find the dataset [here](./dataset/telco_churn_raw.csv).

We start by preparing and cleaning the data. Then we will also explore the topic of [**Multicollinearity**](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) among numerical variables. We will also try to select only those categorical variables with a considerable [**Mutual Information**](https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8) score based on a chosen threshold of 0.001. 

We will also explore Onehot Encoding of **Nominal Variables** using sklearn's [**DictVectorizer**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html).

We will explain Logistic Regression in detail, and interpret the results of our model on one record, so that we are sure that we understand what the results are.

We will assess the performance of our Logistic Model using Classification Metrics, with a primary focus on [**Recall**](https://towardsdatascience.com/precision-recall-and-f1-score-of-multiclass-classification-learn-in-depth-6c194b217629), and the [AUC or Area Under the Curve and the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) since this metric is useful for binary classification problems.

To make sure you install the appropriate libraries, please run:

`pip install requirements.txt`

I used Streamlit to build a simple application for this repository. This was inspired from this [blog post](https://neptune.ai/blog/how-to-implement-customer-churn-prediction). 

Make sure that you create a `.streamlit` directory which will contain TOML files such as `config` and `credentials` that Streamlit will look for to run the app.

To run the application:

`streamlit run telco_churn_prediction_app.py`

The application can process a customer individually or by batch. When using Batch process, the user can download first a zip file that contains a [csv template](./template/customer_template.csv) which he/she can use to fill up the information about his customers. It also includes a [text file](./template/customer_template.txt) that explains the restrictions that need to be followed for each category of a customer.

Here are some snapshots of the application:

![1](https://github.com/parenriquez/telco-customer-churn-prediction/assets/105270881/93d70741-faf9-430b-a469-82e7a069591a, width="400")

![2](https://github.com/parenriquez/telco-customer-churn-prediction/assets/105270881/06560e1e-1082-4ef6-b937-96e6a7793b9b, width="400")

![3](https://github.com/parenriquez/telco-customer-churn-prediction/assets/105270881/9f07774a-eddc-4834-b53b-3e88130fcc14, width="400")

Hope you will learn many things through this repository about Logistic Regression and Classification problems in general as much as I did. The resources which have been of great help in my learning and building this repository are linked in this README and you can just click them to explore on your own.

#### **Thank You!**
