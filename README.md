## Using RapidMiner GO with Tableau Products
This projects walk thru examples of leveraging RapidMiner's Go Engine for AutoModel and Tableau Prep and Tableau Studio and Server
The common steps for using the solution are as below, followed by examples of integration with Tableau Desktop, Tableau Prep   

1) Install rapidminer-go-python package by running the following command in conda our your preferred python environment

pip install rapidminer-go-python

2) Install tabpy by following instructions here https://github.com/tableau/tabpy

3) Download the following controller files from this project. Depending on what you are trying to do, you may not need all of the following methods



      a. QuickModelTrainingController.py
  
        Used when building predictive models from TableauPrep and Tableau desktop, with minimal settings
  
      b. ScoreController.py
        Used when getting predictions from a model that is deployed on RapidMiner GO, when you want to consume in Tableau Prep /Studio or Server
  
      c. TrainController.py
      
        Used when building predictive models from TableauPrep and Tableau desktop, with minimal settings
  
  

    
