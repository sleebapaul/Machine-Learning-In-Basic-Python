# ML In bare Python 

This is something I wanted to do a long time ago. ML in basic Python. 
No fancy frameworks, just basic Python data structures to code up simple ML algos like Linear Regression.
I didn't use even `numpy` since, ... let's do it in the hard way :smiley: 
Also, we'll appreciate `numpy` only we'll get it know what happens in its absence.
The code is well commented, I'm providing the links to understand the theory.

Many might be having an opinion that the code is less efficient and verbose, but that is not the point here. 
The ideal audience are experienced developers who don't touch ML thinking that it requires a lot of framework meta learning. 
You may start from here, understand what happens in the ground level, appreciate the work already done and then start using frameworks.

Most importantly, here you'll understand how an ML algorithm works exactly and what are the basic building blocks of it. This helps a lot in applied ML since we get to know where to tweak and how to tweak.

This is reinventing the wheel? Yes. But sometimes we learn a lot if we start from the scratch.

## Linear Regression using basic Python data structures 

Dataset used can be found [here](https://drive.google.com/file/d/1fiHg5DyvQeRC4SyhsVnje5dhJNyVWpO1/view). 
I only used a part the entire dataset. Find the data in `weatherData` folder.

### Reference and Theory

[1] [https://www.geeksforgeeks.org/linear-regression-python-implementation/](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

[2] [https://realpython.com/linear-regression-in-python/](https://realpython.com/linear-regression-in-python/)

[3] [https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)
 
[4] [https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f](https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f)

[5] [https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html)

## Logistic Regression using basic Python data structures 

Dataset used can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris). 
I only used a part the entire dataset. Find the data in `irisData` folder.

### Reference and Theory

[1] [https://github.com/KnetML/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section3-logistic-regression.ipynb](https://github.com/KnetML/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section3-logistic-regression.ipynb)  

[2] [https://blog.goodaudience.com/logistic-regression-from-scratch-in-numpy-5841c09e425f](https://blog.goodaudience.com/logistic-regression-from-scratch-in-numpy-5841c09e425f)  

[3] [https://en.wikipedia.org/wiki/Logistic_function](https://en.wikipedia.org/wiki/Logistic_function)  

[4] [https://github.com/leventbass/logistic_regression/blob/master/Logistic_Regression.ipynb](https://github.com/leventbass/logistic_regression/blob/master/Logistic_Regression.ipynb)  


## Naive Bayes Classifier using basic Python data structures 

Dataset used can be found [here](https://www.geeksforgeeks.org/naive-bayes-classifiers/). 
Find the data in `golfData` folder.

NB: This works only for categorical features, not continuous features. We may need algorithms like Gaussian Naive Bayes, 
for handling continuous features. 

### Reference and Theory

[1] [https://www.hackerearth.com/blog/developers/introduction-naive-bayes-algorithm-codes-python-r](https://www.hackerearth.com/blog/developers/introduction-naive-bayes-algorithm-codes-python-r)  

[2] [https://www.geeksforgeeks.org/naive-bayes-classifiers/](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

[3] [https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0](https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0)





