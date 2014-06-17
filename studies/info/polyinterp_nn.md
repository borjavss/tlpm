Simple Artificial Neural Network to build a Polynomial Interpolation
====================================================================

### Introduction

> In the mathematical field of numerical analysis, **interpolation** is a method of constructing new data points within the range of a discrete set of known data points.

![Curve fitting 1](http://upload.wikimedia.org/wikipedia/commons/0/0a/Curve_fitting.png "Fitting curve. Image 1")
<p><span>Polynomial curves fitting points generated with a sine function<br /><font color="red">Red</font>: 1st degree. <font color="green">Green</font>: 2nd degree. <font color="orange">Orange</font>: 3rd degree. <font color="blue">Blue</font>: 4th degree polynomial.</span></p>

> In engineering and science, one often has a number of data points, obtained by sampling or experimentation, which represent the values of a function for a limited number of values of the independent variable. It is often required to **interpolate** (i.e. estimate) the value of that function for an intermediate value of the independent variable. This may be achieved by curve fitting or regression analysis.

![Curve fitting 2](http://upload.wikimedia.org/wikipedia/commons/a/a8/Regression_pic_assymetrique.gif "Fitting curve. Image 2")
<p><span>Fitting of a noisy curve by an asymmetrical peak model, with an iterative process (Gauss Newton algorithm with variable damping factor α).<br /> Top: raw data and model.Bottom: evolution of the normalised sum of the squares of the errors.</span></p>

> A problem closely related to interpolation is **the approximation of a complicated function by a simple function**. Suppose the formula for some given function is known, but too complex to evaluate efficiently. A few known data points from the original function can be used to create an interpolation based on a simpler function.

> Although we tend to think that using a simple function drives us to more interpolation errors, depending on the problem domain and the interpolation method used, the gain in simplicity is typically better than the resultant loss in accuracy.

### Polynomial interpolation

In numerical analysis, **polynomial interpolation** is the interpolation of a given data set by a polynomial: given some points, find a polynomial which goes exactly through these points.

And, this is what we are going to do, we want to get our polynomial function based on these points:

<table border="1" style="width:300px">
<tr align="center">
<td>1 </td><td>2 </td><td>3</td><td>4</td><td>5</td><td>6</td>
<td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td>
<td>13</td><td>14</td><td>15</td><td>...</td><td>25</td>
</tr>
<tr align="center">
<td>0 </td><td>5 </td><td>25</td><td>60</td><td>110</td><td>180</td>
<td>275</td><td>400</td><td>555</td><td>671</td><td>750</td><td>790</td><td>810</td><td>810</td><td>810</td><td>...</td><td>810</td>
</tr>
</table>

based on a **power conversion curve** of a wind turbine (800kW), [ENERCON E48](http://www.enercon.de/p/downloads/EN_Productoverview_0710.pdf).

Advanced methods like [Newton interpolation](http://en.wikipedia.org/wiki/Polynomial_interpolation) or, directly, [Lagrange polynomials](http://en.wikipedia.org/wiki/Lagrange_polynomial) could be used to get this goal. 
Nevertheless, we are going to create a simple **artificial neural network** (ANN) using R language and *nnet* package to get this polynomial interpolation.
### Unique hidden layer neural network to polynomial interpolation
Some critics says [Artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) (ANN) are more complex than simulators and they require more learning to fully operate and are more complicated to develop. We are going to show it as a simple example without do comparisons.

### 1) Taking the curve points that we have and, we plot it to see if it corresponds with our expectations.
```{r init, fig.width=7, fig.height=4}
rm(list=ls()) #eliminating R environment previous values
powererror<-1 #initializing one variable
x<-c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)
y<-c(0,0,5,25,60,110,180,275,400,555,671,750,790,810,810,810,810,810,810,810,810,810,810,810,810)
plot(x,y, col="blue", lwd=4, xlab="",ylab="") #plotting our data points
lines(x,y, col= "green", lwd=6) #plotting future desired line approximation
title(main="Power curve given by their points - Enercon E48 -", col.main="black", font.main=4)
title(ylab="Power (kW)"); title(xlab="Wind speed (m/s)")
```
<p><span><font color="blue">Blue</font> points are the given data (see previous table) and <font color="green">green</font> line is the representation of our desired interpolation curve.</span></p>

### 2) Creating our artificial neural network by learning with the given points of the table
```{r learning}
library(nnet) # we call the nnet CRAN library (it should be installed)
for(i in 1:100){ # we start a 100 iteration
   temp<-nnet(x/20,y/800, size=8, linout=T, maxit=10000, abstol=1.0e-8, reltol=1.0e-12)
      if(temp$value < powererror) {
         powerfct <- temp
         powererror <- powerfct$value
      }
}; rm(i)
```
Altought we start a 100 iteration for learning, it has a condition to stop the proccess when we get our proposit.
Note two things:
- Values are previously normalized
- This ANN has only one hidden layer with 8 neurons in this one.

### 3) Creating our regression function by learned artificial neural network
```{r function}
potencia <- function (velocidad){ # the income is the wind speed
   potencia <- rep(0, length(velocidad)) #initializing power
   for(i in 1:length(velocidad)){ # loop to see in what interval is the value
      if((velocidad[i]>2) & (velocidad[i]<14)) #if wind speed is in correct interval
         potencia[i] <- 800*predict(powerfct,velocidad[i]/20) #applying the ANN
      if(velocidad[i]>=14) # if wind speed is outside desired interval
         potencia[i]<-810
   }
   potencia # the returned value of the function
}
```
Once artificial neural network (ANN) is already created in previous point, the function function to convert **wind speed** into **power** is written. Note that we de-normalized the ANN to get real values.

### 4) Testing if results are as we desire
```{r test, fig.width=4, fig.height=4}
y_predicted<- potencia(x)
plot(y_predicted,y, col=rainbow(6), pch=3, lwd=15, ylab="Predicted Power (kW)", xlab="Given Power (kW)")
```
Always, all results should be tested. Remember that x is the vector of the wind speed values given by the previous data table.
So, if we call the function created on point 3, by passing our x values (wind speed in m/s) we obtain our predicted power values.
To compare it, we get plot **predicted power** in face of given **effective power** by data table. By obtaining a perfect straight line, we could claim perfect correlation.

To plot our new **predicted power** curve created by ANN, you can use the following chunk code:
```{r test_plot, fig.width=7, fig.height=4}
plot(x,potencia(x), type="l", col="red",lwd=12, ylab="Power (kW)", xlab="Wind speed (m/s)")
```
The red curve show the results.

### Conclusion
Artificial Neural Networks (ANN) could serve to create a polynomial interpolation by this prediction and classification power given on many intelligence artificial application in recent days.
