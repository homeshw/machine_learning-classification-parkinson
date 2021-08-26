----------------------------------------------------------------
----------------- README - ASSESSMENT 3 ------------------------
----------------------------------------------------------------
Dataset

https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings

Solution comprises 2 files

1. visualize_data.py - to analyze and visualize data for reporting purposes
2. classificationmodel.py - to build the classification models for each test cases and generate and save relavent metrices

---------- results and visualizations - save location -----------------------------
2 folders will be created (if not exist) to save plots and csv files, except classificatiodata.csv
- all the visualization results will be saved under visualization folder
- all the plots and martices generated by each test case model will be saved to results folder

---------- classificationmodel.py - runtime instructions ----------------------------
- for each test case, the model will run 'number_of_iterations' times (defined as 'number_of_iterations' in the source code). Reports has been done setting number_of_iterations=10. But because of resource issues, number_of_iterations is set to 1 while submitting to Ed
- code comprise a main() function, a NeuralNet class and funtions to create,run and evaluate each test case
- all the test cases will be run while executing. To cotrol any test case not to run, simply comment the test case in main() function. (for example test3 has been commented in below example)
    test1(nn)
    test2(nn)
    #test3(nn)
    test4(nn) 
- code will use train.txt and test.txt (original data sets) for model training and validations.
- a preprossessed dataset will be generated as classificatiodata.csv
