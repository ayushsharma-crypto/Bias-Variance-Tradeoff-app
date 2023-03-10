# Bias-Variance-Tradeoff
A python code for demonstration of Bias-Variance TradeOff concept in Machine Learning


### By
	Ayush Sharma - 2019101004
	Nitin Chandak - 2019101024

**NOTE:** Refer `code.pdf` for better explanation & understanding

# Observation & Report

LinearRegression().fit()
-------------------------

Linear Regression is a supervised machine learning algorithm that is used to predict values in a continuous range. LinearRegression().fit(x,y) is a function of the Class sklearn.linear_model.LinearRegression. LinearRegression().fit(x, y) basically fits the model to the training dataset during the training part of the process. It helps to find the coefficients for the polynomial equation(using the concept of gradient descent) which then we will be using to predict the output for the test dataset. LinearRegression().fit(x, y) returns self, which is an instance of the class Linear Regression.


Tabulating Bias & Variance for each Polynomial
----------------------------------------------

### BIAS

As we know that Bias is the error, which is the difference between the average prediction of our model and the correct value which we are trying to predict. If our trained model is more inclined towards underfit situation then Bias will be high. On the otherhand in case of Overfit situation bias will be low.

From the below tabulated data of Bias for each Degree we can see as complexity of function increases (Degree here), it fits better in the Training Dataset. Later in the below section one can find Bias^2 vs Degree Graph & observe a steadily decreasing trend in the Graph upto polynomial of degree 3. The Bias starts increasing after 3rd degree polynomial which refers to the fact that Cubic Polynomial fits best for the given Training Data and Test Data. And further polynomial won't have any serious purpose or value.

### VARIANCE

Variance refers to the variability of a model prediction for a given data point. We can see general increase in the value of variance as complexity of function(degree here) increases. This is because as functional complexity increases, the predicted function becomes more prone to minor changes in the training or testing dataset. This will be reflected in the predicted co-efficients of the predicted function. Leading to high variance on the dataset.

Tabulating Irreducible Error for each Polynomial
------------------------------------------------

### IRREDUCIBLE ERROR

An irreducible error is an error that you get not because your model is not correct, but because of the noise in the data you are training or testing on. Hence, irreducible error doesn't change much with the model i.e our polynomial models from degree 1 to 20. The order of irreducible error is of  10^(???10) which is very small and can't be reduced .And the negative values of irreducible error are due to the floating-point precision error of the python interpreter.


### Plotting Bias^2 ??? Variance Trade-Off graph

As we can see from the below graph that Bias square drastically decreases while going from Quadratic to Cubic Polynomial due nature of test data. In short, our test data resembles Cubic polynomial with some noise added to it. We can observe Degree 1 & 2 models are Underfit for the given test/train dataset as their Bias or Bias Square is relatively much higher as compared to polynomial of degree greater than equal to 3. For later polynomial i.e. degree >= 3 bias remains approximately same but variance consistently increases. Leading to high variance. Thus Overfit model.
