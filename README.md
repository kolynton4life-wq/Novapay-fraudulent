# Novapay-fraudulent - Real‑Time Fraud Detection

## Overview

NovaPay Fraud is a real-time fraud detection project created to improve the safety and reliability of digital financial transactions across web, mobile, and ATM channels. In today’s fast-moving payment environment, transactions happen within seconds, which means suspicious activity must also be identified just as quickly. This project was designed with that need in mind. It uses historical transaction records, behavioral patterns, and carefully engineered features to help detect unusual activity that may indicate fraud.

The main goal of NovaPay FraudGuard is to support early detection rather than reacting after damage has already occurred. By studying how customers normally behave, how transactions vary across channels, and how activity changes by time of day, the system can highlight transactions that appear inconsistent or risky. This makes it possible to flag potential fraud in real time and reduce the chance of financial loss.

In addition to improving fraud detection, the project also aims to strengthen trust in the payment platform. When customers feel that their transactions are being monitored intelligently and securely, they are more likely to use the service with confidence. For that reason, NovaPay FraudGuard is not only a technical solution but also an important step toward building a safer and more dependable digital payment experience.

## Problem Statement

Fraud continues to be one of the most serious challenges facing digital payment systems. It affects businesses, customers, and financial institutions by causing direct monetary losses, increasing operational costs, and weakening trust in the platform. Even a small number of fraudulent transactions can create a major impact when transaction volumes are high, which is why early detection is so important.

An analysis of the NovaPay transaction dataset shows that approximately **9%** of all recorded transactions are fraudulent. While this may seem like a relatively small proportion, it represents a meaningful risk when considered across a large number of daily transactions. The data also shows that fraud is not evenly spread across all transactions. Instead, it appears more frequently in certain channels, especially web-based transactions, and during specific time periods, particularly in the early morning hours.

These patterns suggest that fraud is influenced by more than just random chance. Transaction channel, timing, and customer behavior all appear to play a role in how fraudulent activity occurs. This makes manual review alone too slow and too limited to handle the problem effectively. As digital transactions continue to grow, there is a clear need for a smarter and more proactive approach.

A data-driven fraud detection system is therefore necessary to identify suspicious transactions as they happen, support faster decision-making, reduce financial exposure, and improve the overall security of the payment platform. By detecting risk earlier, NovaPay FraudGuard helps protect both the business and its customers while creating a more trustworthy transaction environment.

# Specific Objectives

The primary objective of the NovaPay Fraud project is to develop a robust and intelligent fraud detection pipeline capable of identifying suspicious financial transactions in real time. The system is designed to analyze transaction data, recognize abnormal behavioral patterns, and distinguish fraudulent activities from legitimate transactions with a high level of accuracy. Achieving this objective will help reduce financial losses, improve operational efficiency, and strengthen customer confidence in digital payment services.

To accomplish this goal, the project focuses on the following specific objectives:

* **Develop a real-time fraud detection framework** capable of automatically identifying suspicious transactions before they result in financial losses.

* **Analyze historical transaction data** to uncover fraud patterns across different transaction channels, countries, customer groups, and time periods. This analysis provides valuable insights into when, where, and how fraudulent activities are most likely to occur.

* **Engineer meaningful predictive features** such as transaction frequency, account age, chargeback history, customer spending behavior, and transaction velocity. These engineered variables provide additional information that improves the ability of machine learning models to distinguish fraudulent transactions from legitimate ones.

* **Perform comprehensive exploratory data analysis (EDA)** to understand the structure of the dataset, identify trends, detect anomalies, and examine relationships between variables before model development.

* **Conduct statistical and correlation analysis** to identify the strongest predictors of fraud and understand the relationships among numerical and categorical variables.

* **Create informative data visualizations** that clearly communicate fraud patterns, customer behavior, transaction trends, and model insights, enabling stakeholders to make informed business decisions.

* **Prepare a clean and reliable analytical dataset** by removing duplicate records, handling missing values, correcting inconsistent data, and ensuring the dataset is suitable for predictive modeling.

* **Support future machine learning model development** by providing a well-structured dataset with high-quality engineered features that improve model performance and prediction accuracy.

---

# Data Dictionary

The NovaPay Fraud dataset contains transaction-level information collected from multiple digital payment channels. Each row in the dataset represents a single financial transaction together with customer information, transaction characteristics, and the fraud label used for predictive modeling.

| **Column**         | **Description**                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **transaction_id** | A unique identifier assigned to each transaction to ensure every transaction can be individually tracked.          |
| **customer_id**    | A unique identifier assigned to each customer performing transactions on the platform.                             |
| **timestamp**      | The exact date and time when the transaction occurred. This field is used to generate several time-based features. |
| **channel**        | The platform through which the transaction was initiated, such as Web, Mobile, ATM, or Unknown.                    |
| **home_country**   | The country where the customer is officially registered.                                                           |
| **currency**       | The currency used to perform the transaction.                                                                      |
| **fee_clean**      | The cleaned transaction fee after preprocessing and data cleaning.                                                 |
| **is_fraud**       | The target variable indicating whether a transaction is fraudulent (1) or legitimate (0).                          |

---

# Engineered Time Variables

To improve fraud detection performance, several additional time-related variables were extracted from the original timestamp column. These engineered features help identify temporal fraud patterns that may not be immediately visible in the raw dataset.

| **Feature**               | **Description**                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **hour**                  | The hour (0–23) during which the transaction occurred. Useful for identifying periods with higher fraud activity.                                            |
| **day_name**              | The day of the week (Monday to Sunday) on which the transaction occurred.                                                                                    |
| **month**                 | The calendar month (1–12) of the transaction.                                                                                                                |
| **year**                  | The year in which the transaction occurred.                                                                                                                  |
| **is_weekend**            | Indicates whether the transaction occurred during the weekend (1) or on a weekday (0).                                                                       |
| **transactions_last_24h** | The total number of transactions completed by the same customer within the previous 24 hours. This feature helps detect unusually high transaction activity. |

---

# Target Variable

The **is_fraud** column serves as the target variable for the fraud detection model. It represents the outcome that the machine learning algorithms are trained to predict.

| **Value** | **Meaning**            |
| --------- | ---------------------- |
| **1**     | Fraudulent Transaction |
| **0**     | Legitimate Transaction |

This binary classification variable enables the model to learn the characteristics that distinguish fraudulent transactions from genuine customer activities.

---

# How to Interpret a Single Transaction

Each row in the dataset represents one completed financial transaction together with the customer's profile and transaction characteristics. By examining the values contained in a single row, analysts can understand the context of the transaction and determine whether it exhibits suspicious behavior.

### Example Transaction

| **Feature**         | **Value**              |
| ------------------- | ---------------------- |
| Channel             | Web                    |
| Home Country        | United States          |
| Hour                | 15 (3:00 PM)           |
| Account Age         | 120 Days               |
| Chargeback History  | 2 Previous Chargebacks |
| Internal Risk Score | 0.78                   |
| Fraud Status        | 1 (Fraud)              |

### Interpretation

This transaction was initiated through the **Web** channel by a customer registered in the **United States**. The customer's account has been active for **120 days** and has previously recorded **two chargebacks**, indicating a history of disputed transactions. The transaction occurred at **3:00 PM** and received a relatively high internal risk score of **0.78**, suggesting elevated fraud risk. Based on these characteristics, the transaction was classified as **fraudulent (is_fraud = 1)**.

This example demonstrates how multiple transaction attributes collectively contribute to fraud detection rather than relying on a single indicator.

---

# Data Cleaning and Preprocessing

Data quality is one of the most important factors influencing the performance of any machine learning model. Before conducting exploratory data analysis and model development, the dataset was thoroughly cleaned to improve consistency, accuracy, and reliability.

## Duplicate Record Detection

Duplicate transactions were investigated by examining the **transaction_id** field, which uniquely identifies every transaction.

The following steps were performed:

* Checked the dataset for duplicate transaction identifiers.
* Identified duplicate records resulting from repeated entries.
* Removed duplicate observations to ensure that each transaction appeared only once.
* Verified the integrity of the dataset after duplicate removal.

Removing duplicate records prevents biased statistical analysis and ensures that machine learning models are trained using accurate information.

---

## Missing Data Investigation

The dataset was examined for missing values across all numerical, categorical, and datetime variables.

The investigation included:

* Identifying variables containing missing values.
* Calculating the percentage of missing observations.
* Evaluating the impact of missing data on the overall dataset.
* Determining the most appropriate treatment strategy for each variable.

Special attention was given to important variables such as **currency**, **transaction fee**, **IP address**, **device trust score**, and other transaction-related attributes.

---

# Handling Missing Values

Different imputation techniques were applied depending on the type and importance of each variable.

### Numerical Variables

Missing numerical values were replaced using the **median** instead of the mean. The median is less sensitive to extreme values and outliers, making it more suitable for financial transaction data.

### Categorical Variables

Missing categorical values were replaced using the **most frequently occurring category (mode)**. Where appropriate, missing categories were labelled as **"UNKNOWN"** to preserve useful information without introducing bias.

### Datetime Variables

The **timestamp** column was converted into a standardized datetime format. Records containing invalid or unreadable timestamps were removed because accurate time information is essential for generating temporal features.

### Currency Information

Missing currency values, particularly transactions involving **USD**, were carefully investigated. Depending on the completeness and reliability of the available information, missing entries were either imputed using the most appropriate category or removed when they could not be reliably recovered.

### Fraud Labels

The target variable (**is_fraud**) was validated to ensure that no missing values were present. Since this variable represents the prediction target, maintaining complete fraud labels is critical for successful machine learning model training.

---

# Data Cleaning Summary

The data preprocessing stage resulted in a clean, consistent, and reliable dataset suitable for exploratory data analysis, feature engineering, and predictive modeling. Key preprocessing activities included:

* Removal of duplicate transaction records.
* Detection and treatment of missing values.
* Standardization of datetime formats.
* Creation of new temporal features.
* Validation of fraud labels.
* Preparation of a high-quality analytical dataset for subsequent machine learning tasks.

These preprocessing steps improve the quality of the data, reduce potential bias, and enhance the performance of the fraud detection models developed in later stages of the project.

