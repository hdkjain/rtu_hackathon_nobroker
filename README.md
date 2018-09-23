# rtu_hackathon_nobroker
Design a machine learning algorithm to predict the rent of a property.  You will have following days available  100,000 property data from a particular City. Will have around 50 attributes of the property Data of the neighborhood of these properties You are free to source more data as needed.  Given a new property your algorithm should be able to predict a rent range of the property.

#EVALUATION OF TOP-10 HEAT MAP
 Please go through the presentation “#itout[1].pptx” for brief introduction about how we calculated the top 10 attributes and  go through the script “Top _10_Attribute.py”

IMPORT DATA :  1.test.csv
         		  2. train.csv

Result we got:

1 
Bathroom 
2 
Type(x-BHK) 
3 
Deposit 
4 
Lift 
5 
Property Size 
6 
Gym 
7 
Pool 
8 
Security 
9 
Total-Floor 


RESULT PREDICTION
To predict the rent of a new property with the help of given attributes run
“HACKATHON.PY” ,“NEWDATA .CSV”
In these scripts we have compared various models such as Linear Regression , Multiple Regression and Polynomial Regression to viualize and analyse which suits best for our data set . 
In our Experiment we have received a accuracy of 75% through Polynomial Regression model.

BASE-VALUE  OF EVERY LOCALITY
To create a base value of rent for a locality we tried to use K nearest Neighbour Clustering in which we only used the “latitude, longitude” and “rent”  attributes of the data available to cluster the data and get a mean/base value for every locality.
