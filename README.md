# Machine-Learning-Algorithms-in-CPP
by Hamna Mustafa and Sanika Buche

We implemented Logistic Regression and Naive Bayes from scratch in C++ on the titantic dataset and then compared the results and performance to the equivalent functions in R


# Analysis
# Logistic Regression
The accuracy for our logistic regression was 67%. This is considered a good accuracy because it is closer to 100% than it is to 0%. We got the same coefficients and metrics with both our R code as well as our C++ code as can be seen by the attached screenshots.
The runtime for our C++ code was 108.89 seconds and the runtime for our R code was 0.042 seconds. The runtime for C++ and R is different because C++ is a compiled language while R is an interpreted language. We used proc.time() to calculate the R runtime by using it at the start and end of the machine learning part of the code and then subtracting the difference. We measured the C++ runtime by using the high_resolution_clock in chrono to get the start time and end time of the machine learning part of the code and then subtracted the difference.

Logistic Regression in R:

<img width="468" alt="image" src="https://user-images.githubusercontent.com/42907026/147741387-0bbd15f4-ac71-4344-82dc-229e71a88470.png">

<img width="392" alt="image" src="https://user-images.githubusercontent.com/42907026/147741402-7adb1a62-d0bd-412f-b0fd-46106610263e.png">

Logistic Regression in C++:

<img width="419" alt="image" src="https://user-images.githubusercontent.com/42907026/147741445-f7997aad-b216-4395-8051-56feb0feba4e.png">

# Naive Bayes

The accuracy for our Naive Bayes was 76%. This is considered a good accuracy because it is closer to 100% than it is to 0%. We can see that this was a better model than the Logistic Regression model because it had a better accuracy. That may have been because we used more predictors in this model than in the previous one. We got the same conditional probabilities and metrics with both our R code as well as our C++ code as can be seen by the attached screenshots.
The runtime for our C++ code was 0.001 seconds and the runtime for our R code was 0.007 seconds.
The runtime for C++ and R is different because C++ is a compiled language while R is an interpreted language. We used proc.time() to calculate the R runtime by using it at the start and end of the machine learning part of the code and then subtracting the difference. We measured the C++ runtime by using the high_resolution_clock in chrono to get the start time and end time of the machine learning part of the code and then subtracted the difference. 

Naive Bayes in R:

<img width="395" alt="image" src="https://user-images.githubusercontent.com/42907026/147741516-bc112ba4-2243-4516-9ec8-3b082e4eb48a.png">

<img width="401" alt="image" src="https://user-images.githubusercontent.com/42907026/147741527-892469ab-cebf-4dab-8d67-3c839bfbf100.png">

Naive Bayes in C++:

<img width="483" alt="image" src="https://user-images.githubusercontent.com/42907026/147741554-ac6288a8-1c42-417d-9980-3c8a9b836fc0.png">
