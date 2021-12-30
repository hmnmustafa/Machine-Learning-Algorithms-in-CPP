//  Hamna Mustafa hbm170002
//  Sanika Buche ssb170002
//
//  main.cpp
//  ML-HW5-NaiveBayes
//
//  Created by Hamna Mustafa on 10/1/21.
//  Copyright © 2021 Hamna Mustafa. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;

//this function calculates the total number of 0s and 1s in a vector
vector<double> findLength (vector<double> v1) {
    vector<double> result(2);
    
    double sumTrue = 0;
    double sumFalse = 0;
    for (int i = 0; i< v1.size(); i++){
        if (v1[i] == 1)
            sumTrue++;
        else if (v1[i] == 0)
            sumFalse++;
    }
    result[0] = sumFalse;
    result[1] = sumTrue;
    return result;
}

//this function calculates the likelihood of a continious predictor (age, in our case)
double calculateAgeLh (double age, double mean, double var){
    double ageLh = (1 / sqrt(2* (22/7) * var)) * exp(-(pow(age-mean,2))/(2*var));
    return ageLh;
}

//This function calculates the true positives, true negatives, false positives and false negatives of the model. It takes a vector of predictions and a vector of actual values and compares the two to find the above metrics
vector<double> findMetrics(vector<double> v1, vector<double> v2) {
    double TP = 0;
    double FP = 0;
    double TN = 0;
    double FN = 0;
    
    for (int i = 0; i<v1.size(); i++){
        if (v1[i]==v2[i]){
            if (v1[i]==1)
                TP++;
            else if (v1[i] == 0)
                TN++;
            
        }
        else {
            if (v1[i]==1)
                FP++;
            else if (v1[i] == 0)
                FN++;
        }
    }
    vector<double> result = {TP,TN,FP,FN};
    return result;
}

int main(int argc, const char * argv[]) {
    ifstream inFS;
    string line;
    string trash_in, pclass_in, survived_in, sex_in, age_in;
    const int  MAX_LEN = 1046;
    vector<double> pclass(MAX_LEN),survived(MAX_LEN), sex(MAX_LEN), age(MAX_LEN);

    
    cout << "Opening file titanic_project.csv" << endl;
    inFS.open("titanic_project.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file titanic.csv" << endl;
        return 1; //1 indicates error
    }
    else {
        cout << "File opened"  <<endl;
    }
    
   // cout << "Reading line 1" << endl;
    getline(inFS, line);
  //  cout << "heading: " << line  << endl;
    
    
    
    int numObservations = 0;
    while (inFS.good()) {
         
         //reading in the data
         getline(inFS, trash_in,',');
         getline(inFS, pclass_in, ',');
         getline(inFS, survived_in, ',');
         getline(inFS, sex_in, ',');
         getline(inFS, age_in, '\n');
         
         pclass.at(numObservations) = stof(pclass_in);
         survived.at(numObservations) = stof(survived_in);
         sex.at(numObservations) = stof(sex_in);
         age.at(numObservations) = stof(age_in);
         
         numObservations++;
     }
     
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);
    
    //creating the train and test set
    vector<vector<double> > train(4, vector<double> (900));
    vector<vector<double> > test(4, vector<double> (146));

    //filling first 900 observations of the data set in the train data matrix
    for (int i=0; i<900; i=i+1) {
        train.at(0).at(i) = pclass.at(i);
        train.at(1).at(i) = sex.at(i);
        train.at(2).at(i) = age.at(i);
        train.at(3).at(i) = survived.at(i);
    }

    //filling in remaining observations of the data set in the test data matrix
    int j = 0;
    for (int i=900; i<=pclass.size()-1; i++) {
        test.at(0).at(j) = pclass.at(i);
        test.at(1).at(j) = sex.at(i);
        test.at(2).at(j) = age.at(i);
        test.at(3).at(j) = survived.at(i);
        j++;
    }
    
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    //calculating the counts of survived and didn't survive
    vector<double> count_survived = findLength(train[3]);
    
    //calculating priors
    vector<double> apriori(2);
    apriori[0] = count_survived[0]/train[0].size();
    apriori[1] = count_survived[1]/train[1].size();
    
    //calculating the likelihoods of pclass given survived and not survived
    vector<vector<double> > lh_pclass(3, vector<double> (2));
    for (int sv=0; sv<lh_pclass[0].size(); sv++){
        for (int pc=0; pc<lh_pclass.size(); pc++){
            double sum = 0;
            for (int  k=0; k<train[0].size(); k++){
                if (train[0][k] == pc+1 && train[3][k] == sv)
                    sum++;
            }
            lh_pclass[pc][sv] = sum/count_survived[sv];
        }
    }
    
    //calculating the likelihoods of sex given survived and not survived
    vector<vector<double> > lh_sex(2, vector<double> (2));
    for (int sv=0; sv<lh_sex[0].size(); sv++){
        for (int sx=0; sx<lh_sex.size(); sx++){
            double sum = 0;
            for (int  k=0; k<train[0].size(); k++){
                if (train[1][k] == sx && train[3][k] == sv)
                    sum++;
            }
            lh_sex[sx][sv] = sum/count_survived[sv];
        }
    }
    
    
    vector <double>  ageMean = {0,0} ;
    vector <double> age_var = {0,0} ;
   
    
    //calculating mean of age
    for (int sv = 0; sv<2; sv++) {
        double sumAge = 0;
        double countAge = 0;
        for (int i=0; i<train[0].size(); i++) {
            if (train[3][i] == sv){
                sumAge+= train[2][i];
                countAge++;
            }
                
        }
        ageMean[sv] = sumAge/countAge;
        
    }
    
    //calculating variance of age
    for (int sv = 0; sv<2; sv++) {
        double sumAge = 0;
        double countAge = 0;
        for (int i=0; i< train[0].size(); i++) {
            if (train[3][i] == sv){
              
                sumAge+= pow((train[2][i] - ageMean[sv]),2);
                countAge++;
            }
                
        }
    
        age_var[sv] = sumAge/countAge;

    }
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    
    vector<vector<double> > probabilities (2,vector<double> (test[0].size()));

    //calculating probabilites of every element in the test set
    for (int i = 0; i<test[0].size(); i++) {
        double num_s = lh_pclass[test[0][i]-1][1] * lh_sex[test[1][i]][1] * apriori[1] * calculateAgeLh(test[2][i], ageMean[1], age_var[1]);
    
        double num_p = lh_pclass[test[0][i]-1][0] * lh_sex[test[1][i]][0] * apriori[0] * calculateAgeLh(test[2][i], ageMean[0], age_var[0]);
 
        
        double denominator = num_s + num_p;
        
        probabilities[0][i] = num_p/denominator;
        probabilities[1][i] = num_s/denominator;

    }
    
    vector<double> predictions (test[0].size());
    
    //converts probability vector into 2 classes where the class is 0 if probability is < 0.5 and 1 if >= 0.5
    for (int i=0; i<probabilities[0].size(); i++){
        if (probabilities[0][i] > 0.5)
            predictions[i] = 0;
        else
            predictions[i] = 1;
    }
    
    
    //displaying conditional probabilities
    cout << endl << "Conditional Probabilites" << endl;

    cout << endl << "pclass     1       2       3" << endl;
    cout << "[0]    " << lh_pclass[0][0] << " " << lh_pclass[1][0] << " " << lh_pclass[2][0] << endl;
    cout << "[1]    " << lh_pclass[0][1] << " " << lh_pclass[1][1] << " " << lh_pclass[2][1] << endl;
    

    cout << endl << "sex         1       2" << endl;
    cout << "[0]    " << lh_sex[0][0] << " " << lh_sex[1][0] << endl;
    cout << "[1]    " << lh_sex[0][1] << " " << lh_sex[1][1]  << endl;

    cout << endl << "age         1       2" << endl;
    cout << "[0]    " << ageMean[0] << " "  << sqrt(age_var[0]) << endl;
    cout << "[1]    " << ageMean[1] << " "  << sqrt(age_var[1]) << endl;
    
    // metrics = {TP,TN,FP,FN};
    vector<double> metrics = findMetrics(predictions, test[3]);
    
    //calculating metrics
    double accuracy = (metrics[0] + metrics[1])/(metrics[0] +metrics[1] + metrics[2] + metrics[3]);
    double  specificity =metrics[0] /(metrics[0] + metrics[3]);
    double sensitivity = metrics[1] /(metrics[1] + metrics[2]);
    
    cout << endl << "Metrics:" << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;
    
    cout<< endl << "Runtime: " << time_span.count() << " seconds." << endl;
    return 0;
}


