This is the GitHub repository for the Applied Machine Learning Systems Assignment at UCL.

This repository contains three different folders. Each folder has one file. Two additoinal files, including the README file are also in the repository.

Folder A contains the file "Pneumonia.py", which includes the class and the code implementation of task A, linked with the PneumoniaMNIST dataset. 

Folder B contains the file "Path.py" which includes the class and the code implementation of task B, linked with the PathMNIST dataset. 

The folder Dataset only includes an empty txt file. GitHub will not allow for an empty folder to exist so the text file was placed there for that reason. Users who want to run this repository must add both Datasets, PneumoniaMNIST and PathMNIST, in their npz file form. The file names should be "pneumonia.npz" and "path.npz". 

The only way to run the code is through the file "main,.py". Once the user runs the code they will be prompted with a menu of whether they want to run task A or task B. For the user to run both tasks two different executions of file "main.py" are needed.

For the execution of the code to work, the user must have the following libraries installed in their Python Environment:

-pip install numpy

-pip install matplotlib

-pip install scikit-metrics

-pip install seaborn

-pip install tensorflow

-pip install keras

