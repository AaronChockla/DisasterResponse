# Udacity Disaster Response Project
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.  The initial data provides labeled tweets from real disaster situations.  The goal of this project is to create an NLP pipeline that accurately categorizes tweets for future disaster situations.  The project is divided into three parts:

**1.** ETL pipeline to clean and process the data

**2.** ML pipeline to train a classifier using clean data

**3.** Web application to visiualize the results


## Dependencies
* **Python:** Python 3+ (I used Python 3.7)
* **Data Analysis Libraries:** NumPy, SciPy, Pandas, Sciki-Learn
* **Other Utility Libraries:** sys, re, pickle
* **NLP Library:** NLTK
* **SQL Library:*** SQLalchemy
* **Web App Libraries:** Flask, Plotly

## Installation

Clone the following repository:

```
git clone https://github.com/AaronChockla/DisasterResponse
```


## Executing Program:
Run the following commands in the project's root directory to set up your database and model.

**1.** To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

**2.** To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

**3.** Run the following command in the app's directory to run your web app. python run.py

**4.** Go to http://0.0.0.0:3001/

## License
This project is licensed under the MIT License - see the [LICENSE] (https://github.com/AaronChockla/DisasterResponse/blob/master/LICENSE) file for details.

## Acknolwedgements
* [Udacity](https://www.udacity.com/)
* [Figure Eight] (https://www.figure-eight.com/) for providing the data set
