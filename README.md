This is the GitHub repository for the Applied Machine Learning Systems Assignment at UCL.

This repository contains three different folders each with one file and two files outside the folders, including the README file. 

Folder A contains the file "Pneumonia.py", which includes the class for the model training in task A, using the PneumoniaMNIST dataset. 

Folder B contains the file "Path.py" which inclused the class for the model training in task B, using the PathMNIST dataset. 

The folder Dataset only includes an empty txt file. GitHub will not allow for an empty folder to exist so the text file was placed there for that reason. Users who want to run this repository must add both Datasets, PneumoniaMNIST and PathMNIST, in their npz file form. The file names should be "pneumonia.npz" and "path.npz". 

The only way to run the code is through the file "main,.py". Once the user runs the code they will be prompted with a menu of whether they want to run task A or task B. 

For the execution of the code to work, the user must have the following libraries installed on their Python Environment:

-pip install numpy
-pip install matplotlib
-pip install scikit-metrics
-pip install seaborn
-pip install tensorflow
-pip install keras

Once the user runs the main file, they will be prompted to run either task A or task B. Therefore, two different executions of the code are required for both tasks to run. 