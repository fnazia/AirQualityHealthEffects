# Association of Air Quality and Mortality Rate Specific to A Disease
Predictive Analysis of air quality effects on human health

#### Datasets:

1. [Air Quality Dataset](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual)
2. [Mortality Rate Dataset](https://ghdx.healthdata.org/us-data)

#### Overview:

Diseases analyzed: Asthma, CRD, COPD, Stroke, Cardiovascular disease <br/>
Data characteristics: Imbalanced dataset of ~1.5GB size <br/>
Measure of human health (label column): Mortality rate <br/>
Measure of air quality (feature columns): 'Good Days',
       'Moderate Days', 'Unhealthy for Sensitive Groups Days',
       'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days', 'Max AQI',
       '90th Percentile AQI', 'Median AQI', 'Days CO', 'Days NO2',
       'Days Ozone', 'Days SO2', 'Days PM2.5', 'Days PM10' <br/>

Predictive algorithm: Random Forest <br/>
Prediction approach: Binary Classification <br/>
Evaluation: Accuracy, Confusion matrix, AUROC curve, Feature ranking 

#### Findings:

Most important feature affecting the mortality rate for - <br/>
Airway diseases (Asthma, COPD, CRD): Days PM2.5 <br/>
Cardiovascular disease: 90th Percentile AQI (other features coming very close) <br/>
Stroke: Good days (other features coming very close) <br/>

![alt text](https://github.com/fnazia/AirQualityHealthEffects/blob/master/imgs/asthma_feature.png?raw=true)

#### Conclusion:

Random Forest could detect PM2.5 as the most important feature responsible 
for the airway diseases affecting human health. This agrees with the 
experimental finding of the impact of PM2.5 on the respiratory system. On 
the other hand, no decisive air-quality feature is detected for 
non-respiratory diseases (Cardiovascular and Stroke) indicating a more 
complex or nonexistent relationship of these diseases with the air quality. 
Therefore, it may be possible to predict the prevalence of an airway disease 
or its effect on the mortality rate by analyzing the air quality data of a 
region.

#### Acknowledgements & References:
  1. https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/
  2. https://www.spotx.tv/resources/blog/developer-blog/exploring-random-forest-internal-knowledge-and-converting-the-model-to-table-form/
  3. https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/RandomForestClassifierExample.scala
  4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4740125/#:~:text=raising%20worldwide%20concerns.-,PM2.,and%20consequently%20impair%20lung%20function.
