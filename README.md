# Implementation-of-Classification-of-Imbalanced-Data-by-Oversampling-in-Kernel-Space-of-SVM-paper
This is the MATLAB implemenation of paper published in  IEEE Transactions on Neural Networks and Learning Systems  Date of Publication: 10 October 2017   ISSN Information: Print ISSN: 2162-237X Electronic ISSN: 2162-2388 PubMed ID: 29028213   DOI: 10.1109/TNNLS.2017.275161
Authors : Josey Mathew, Chee Khiang Pang, Ming Luo, Weng Hoe Leong
Title: Classification of Imbalanced Data by Oversampling in Kernel Space of Support Vector Machines

In order to run main.m which is the actual implementation of above mentioned research paper, you need to setup weighted libsvm installed in you matlab and SMOTE program that will be used to generate synthethic minority class examples.

--> Open matlab folder, use readme inside the folder


--> Install libsvm-weighted package


--> Put code, test dataset(diabetes.csv) and smote code in same directory.


--> Execute "main.m" 


The file will execute the code on diabetes.csv, efficiency of the given code will be near about 82.6%
User need to tune parameter C, Cmin,Cmaj,Csyn yourself.

let me know if there is any bug in code.
