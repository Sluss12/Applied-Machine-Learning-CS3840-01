# Final Project

## Breast Cancer Detection

[Applied Machine Learning CS-3840](https://pilot.wright.edu/d2l/lms/dropbox/user/folder_submit_files.d2l?ou=624767&db=358827)

[Matthew Slusser](https://github.com/Sluss12/Applied-Machine-Learning-CS3840-01/tree/master/final-project)

[Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Instructions

Code implementation: 15
Project report: 20
Slides and presentation: 15

Your final project report should include:

1. Title and author
2. Introduction (background, problem to be solved, and how you want to solve it)
3. Dataset (data statistics, features, how you pre-process your data if there are some, and the sizes of training data and test data)
4. Methods (machine learning methods you used and detailed steps)
5. Experimental results (experimental results using evaluation metrics, including tables and figures)
6. Discussion (give your analysis and observations from experimental results)
7. Challenges and future steps (describe the challenges during the projects, and some possible future steps that can better solve this problem)
8. Conclusion

Please submit your lab report (Firstname-Lastname-final.pdf), code (Firstname-Lastname-final.ipynd), and presentation slides (Firstname-Lastname-final.pptx) separately or in a package (Firstname-Lastname-final.zip) to the dropbox.

---

### Useful links -

- [Breast Cancer Histopathological Database - BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)
- [CBIS-DDSM: Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- [Breast Cancer EDA Notebook for CBIS-DDSM dataset](https://www.kaggle.com/code/awsaf49/breast-cancer-eda/notebook)
- [Breast Can](https://www.kaggle.com/code/yasserh/breast-cancer-diagnosis-best-ml-algorithms)
- [Feature Selection and Data Visualization](https://www.kaggle.com/code/kanncaa1/feature-selection-and-data-visualization)
- [Basic Machine Learning with Cancer](https://www.kaggle.com/code/gargmanish/basic-machine-learning-with-cancer)
- [Classification - Breast Cancer or Not (with 15 ML)](https://www.kaggle.com/code/mirichoi0218/classification-breast-cancer-or-not-with-15-ml)

---

## Table of Contents

- [Table of Contents](matthew-slusser-final.ipynb#table-of-contents)
- [Dataset](matthew-slusser-final.ipynb#data-set)
  - [Dataset Info](matthew-slusser-final.ipynb#display-basic-information-about-the-dataset)
  - [Histograms](matthew-slusser-final.ipynb#plot-dataset-on-histograms)
  - [Preprocess Dataset](matthew-slusser-final.ipynb#preprocess-data-and-separate-the-data-frame-into-features-and-labels)
- [Visualizations](matthew-slusser-final.ipynb#data-visualizations)
  - [Swarm Plots](matthew-slusser-final.ipynb#swarm-plots)
    - [Mean Features](matthew-slusser-final.ipynb#mean-features)
    - [SE Features](#se-features)
    - [Worst Features](matthew-slusser-final.ipynb#worst-features)
  - [Correlation Map](matthew-slusser-final.ipynb#correlation-map)
    - [Mean Features Correlation](matthew-slusser-final.ipynb#mean-features-correlation-map)
    - [SE Features Correlation](matthew-slusser-final.ipynb#se-features-correlation-map)
    - [Worst Features Correlation](matthew-slusser-final.ipynb#worst-features-correlation-map)
- [Models](matthew-slusser-final.ipynb#begin-training-various-models)
  - [Logistic Regression](matthew-slusser-final.ipynb#logistic-regression)
  - [KNN](matthew-slusser-final.ipynb#k-nearest-neighbors)
  - [SVM](matthew-slusser-final.ipynb#support-vector-machine)
  - [Random Forest](matthew-slusser-final.ipynb#random-forest)
  - [Bagging](matthew-slusser-final.ipynb#bagging)
  - [AdaBoost](matthew-slusser-final.ipynb#adaboost)
- [Discussion](matthew-slusser-final.ipynb#discussion)
- [Challenges](matthew-slusser-final.ipynb#challenges)
- [Conclusion](matthew-slusser-final.ipynb#conclusion)
