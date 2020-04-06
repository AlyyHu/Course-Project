# Predicting Pet Adoption Speed for PetFinder(R)

This dataset was provided by **PetFinder**, a non-profit organization, contains a database of animals and aims to improve the animal welfare through collaborations with related parties.The core task of this project is to predict how long it will take for a pet to be adopted. The original dataset can be found on [Kaggle](https://www.kaggle.com/ivotimev/petfinder-adoption-prediction-segmented-1#train_preprocessed.csv). 

We applied **text analytics** in exploratory data analysis and picked ten words that has the highest frequency in the description of each record. Because of our target variable's multi-class characteristic, we choose three models to make prediction, including **Random Forest**, **SVM**, and **Multinomial logistic regression**. We used **LASSO** and **Post-LASSO** to reduce overfitting. We also used **cross-validation** to compare models based on its out-of-sample accuracy. We choose Random Forest as the optimal model with a 65% out-of-sample accuracy.


This is a team project, other creators are [Jiali Yin](https://www.linkedin.com/in/jiali-yin/), [Xinyi Zhu](https://www.linkedin.com/in/xinyi-zhu/), [Linsay Trinh](https://www.linkedin.com/in/lindsay-trinh/), [Shangyun Song](https://www.linkedin.com/in/shangyun-song/) and [Yasi Chen](https://www.linkedin.com/in/yasi-chen-214a8418a/).
