# FEEL THE RHYTHM
## OVERVIEW
You’ll be challenged to build a model that can predict when incidents are more likely to occur, with explainability being the key! 
Because explainability is so important, in this challenge you will be scored on your leaderboard position, and a report that explains your model and your predictions. 

## DESCRIPTION
### Background
The teams at Western Power perform a range of tasks across their operations, and they are looking to better understand what factors contribute to an increased likelihood of safety incidents occurring. 

This challenge is **specifically designed to look at the impact of body clocks, circadian rhythms and work schedules on the likelihood of incidents.**

### Challenge
In this challenge you are asked to build and submit a model that can predict when incidents are more likely to occur, and provide an explanation of your findings and predictions in a report.

> This problem has been framed as hourly classification. For a given employee, working on a given hour, did an incident occur or not? 

### Model Requirements
The winning model is required to deliver the following:

* The model must execute in the Unearthed CrowdML Pipeline which executes within the unearthed/sagemaker-scikit-learn:0.23-1-cpu-py3 Docker container.
* The template provides guidance on where to process data, develop models and make submissions - be careful about changing this.
* A supporting Explainability Report

### Scoring, Assessment and Submitting
#### LeaderBoard
The leaderboard uses Area Under ROC (Receiver operating characteristic) to assess the quality of the models predictions. 

#### Baseline Model
The baseline model is an ExplainableBoostingClassifier model from InterpretML which performs worse than a boosted tree, but is very interpretable.

#### Guidelines for your Explainability Report
Your report should explain your predictions and provide the mechanisms by which your model works. 

Your report may also uncover the biases or recurrent patterns that are present in this dataset, that are influencing the usability or outcomes of your model. Your model explanation might also provide guidance for human decision makers that would be looking to use your model or insights to reduce safety incidents. 

You are not expected to provide explanations for a specific number of records or incidents, but you should highlight key feature influences, with examples of prediction level explanations.

Your explanations should help Western Power understand why your models performed better or worse in some situations than others.

#### What type of insights you might look for?
1. Does it make a difference if employees work day or night shifts?
2. Is there any influence of the season and potential daylight hours throughout the year?
3. Does it make a difference how many days off a person has throughout the year?
4. Does the day of the week make a difference?
5. Are there changes in the effect of work hours related to different incident types or job functions?
6. Think about 'strong' vs 'weak' signals in the data. 

## DATA
You are also allowed to use additional open source data (e.g. circadian rhythm models, weather data), to aid your explanations and insights, but not in training your model. *If you would like to use open source data to train your model, please contact one of our team.*

### Main Challenge Data Set
For the leaderboard, the dataset has been split into training, validation (public leaderboard) and test (private leaderboard). 

The total dataset consists of 20394936 hours worked, 821 of which contain incidents.

> The dataset split was done by employees, and stratified by incidents

Please note that this dataset was not split by time, as causality is not required. Data recording and timesheet processes have also changed overtime. However there are enough incidents to make a model that predicts incidents for unseen employees, and generate model driven insights.

### Training data
12354494 records, each one representing an hour, with 490 of those hours containing incidents. The dataset consists of approximately 60% of records from the overall sample.

### Validation (Public Leaderboard)
We have withheld 3915000 sample records with 156 incidents. These records correspond to unseen employees. The dataset consists of approximately 20% of records from the overall sample.

### Validation (Private Leaderboard)
We have withheld 4125442 sample records with 175 incidents. These records correspond to unseen employees. The dataset consists of approximately 20% of records from the overall sample.

### Data Dictionnary
1. Work_DateTime: The hour for this row
2. EmpNo_Anon: Anonomised Employee identifier
3. Incident_Number: Can be used to tie incidents to the auxiliary data, for your report
4. TIME_TYPE: Overtime or normal
5. WORK_NO: Code for work type
6. WORK_DESC: Full description of work type, e.g. Admin
7. FUNC_CAT: Operational/Support/Network or Asset
8. TOT_BRK_TM: Minutes of break time, averaged per hour in shift

Target - Incident: True if an incident occurred False otherwise

### Understanding the data and data preparation
In this challenge, you are provided with real, but anonymised employee timesheet and incident data. As this is real world data there are some keen points worth considering in this challenge:

1. The timesheet data was originally provided as one row per shift. This has been transformed to create one row per hour of each shift, to allow for hourly predictions.
2. Employees manually enter the amount of break time they had each shift, but not the specific time it was taken. In the hour by hour data, the total break time for the shift, is divided by the number of hours. 
3. The timesheet system and data capture systems have changed slightly over time, and this is evident in the data.
4. Employees manually record the incident data and the time the incidents occurred. 
5. This dataset is from Australia, with southern hemisphere seasons. It can be assumed that most operations occur around the Perth metropolitan area.
6. There are a range of job functions and incident types in the dataset 
7. For employees who logged timesheets but reported no incidents, we assume they experienced no incidents. This means we kept those employees in that data. You may question or alter this assumption in your report
8. For incidents that happened outside of logged hours, we discarded them from the training data, as there was no corresponding timesheet (they are available in the auxiliary data)
9. There are a number of employees who did not use the timesheet system but recorded incidents. They are not present in the training data (they are available in the auxiliary data).