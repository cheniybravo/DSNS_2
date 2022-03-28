# Disaster Response NLP Pipeline
Imagine now there is a sudden natural disaster (or maybe a war) happening somewhere and the affected people are desperately asking for help through sending messages. Now the problem is how can we quickly digest the text messages and send them to the correct aid teams (food, hospital, shelter, etc.)?  
This project solves exactly this problem by firstly applying NLP skills and then buiding a multi-classification model. Final result is visualized in a webpage.


### How to Use:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/MyEngine_Yi.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/MyEngine_Yi.db models/NLP_model_Yi.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to WebPage http://10.88.111.26:3001/

### Files
Below files/folders are contained in this repo which are necessary to run the project.
* `app`: Folder contains all files related to web API:
  1. `run.py`: Code to run this web app.
  2. `templates`: Provided plotly templates used for Web interface.
* data: Folder contains raw data, data processing script and corresponding jupyter notebook:
  1. `disaster_categories.csv`: Raw data.
  2. `disaster_messages.csv`: Raw data.
  3. `process_data.py`: data processing pipeline script.
  4. `ETL_Pipeline_Preparation.ipynb`: Corresponding data processing jupyter notebook.
* models: Folder contains model training script and corresponding jupyter notebook:
  1. `train_classifier.py`: Model training pipeline script.
  2. `ML_Pipeline_Preparation.ipynb`: Corresponding model training jupyter notebook.


### Required Packages
`Numpy`  
`Pandas`  
`sqlalchemy`  
`NLTK`  
`sklearn`  
`re`  
`joblib`   
`sys`  
`flask`  
`plotly`  

### Imbalanced Dataset
The data is imbalanced. If we look at the bubble size below, the event rates of some categories (labels) are as low as 2%. Despite massive discussion over imbalanced data model measurement, the use of precison/recall/f1 score is really, depending on **how likely people are willing to accept False Positive and False Negative in real world**.         

For natural disaster (or maybe a war), False Negative is highly undesirable, as we would not respond to people's true help. On the other side, getting too many False Positives means we are wasting limited aid resources, which is also not cool. 

In such case, F1 score is designed to work well here since it takes into consideration both precision and recall. This project selects model mainly based on f1 score.

![image_info](https://raw.githubusercontent.com/cheniybravo/DSNS_2/master/Img-Volume_EventRate.png)

### Acknowledgements
Data sourced from Figure Eight.