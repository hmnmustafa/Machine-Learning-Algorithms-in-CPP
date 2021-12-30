//  Hamna Musafa hbm170002
//  Sanika Buche ssb170002
//  main.cpp
//  ML-HW5-LogisticReg
//
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

//This function takes in a vector and applies the logistic/sigmoid function on each element in the vector
vector<double> sigmoid(vector<double> v) {
    vector<double> probs(v.size());
    for (int i =0; i<v.size(); i++) {
        probs.at(i) = 1.0/(1+exp(-v.at(i)));
    }
    return probs;
}

//This function multiplies a matrix with a vector
vector<double> multiplication (vector<vector<double>> data_frame, vector<double> weights) {
    vector<double> result (data_frame.at(0).size());
    
    for (int i = 0; i <data_frame.at(0).size() ; i++) {
        for (int j = 0; j < weights.size(); j++){
            result.at(i) += (weights[j]*data_frame[j][i]);
        }
    }
    return result;
}

//This function multiplies the transpose of a matrix with a vector
vector<double> transposeMultiplication (vector<vector<double>> data_frame, vector<double> weights) {
    vector<double> result (data_frame.size());
    for (int i = 0; i <data_frame.size() ; i++) {
        for (int j = 0; j < weights.size(); j++){
            result.at(i) += (weights[j]*data_frame[i][j]);
        }
    }
    return result;
}

//This function subtracts each element of the first vector with the corresponding element of the second vector
vector<double> vectorSubtraction (vector<double> v1, vector<double> v2) {
    vector<double> result(v1.size());
    
    for(int i=0; i<v1.size(); i++) {
        result.at(i) = v1.at(i) - v2.at(i);
    }
    return result;
}

//This function add each element of the first vector with the corresponding element of the second vector
vector<double> vectorAddition (vector<double> v1, vector<double> v2) {
     vector<double> sum(v1.size());
     
     for(int i=0; i<v1.size(); i++) {
         sum.at(i) = v1.at(i) + v2.at(i);
     }
     return sum;
}

//This function takes a vector of predictions and converts them into probabilites so that they all lie between 0 and 1
vector<double> probabilityConversion (vector<double> v) {
    vector<double> prob(v.size());
    for (int i=0; i<prob.size(); i++) {
        prob.at(i) = (exp(v[i]))/(1 + exp(v[i]));
    }
    return prob;
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
    string trash_in, pclass_in, survived_in;
    const int  MAX_LEN = 1046;
    vector<double> pclass(MAX_LEN);
    vector<double> survived(MAX_LEN);
    vector<double> weights(2);
    
    cout << "Opening file titanic_project.csv" << endl;
    inFS.open("titanic_project.csv");
        if (!inFS.is_open()) {
            cout << "Could not open file titanic.csv" << endl;
            return 1; //1 indicates error
        }
        else {
            cout << "File opened"  <<endl;
        }
    

    

    getline(inFS, line);

    
    
    int numObservations = 0;
    while (inFS.good()) {
        
        //reading in the data
        getline(inFS, trash_in,',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, trash_in, '\n');
        
        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        
        numObservations++;
    }
    
    pclass.resize(numObservations);
    survived.resize(numObservations);
    
    fill(weights.begin(), weights.end(),1);
    
    //creating the train and test set
    vector<vector<double>> train(2, vector<double> (900));
    vector<vector<double>> test(2, vector<double> (146));
    
    //filling first 900 observations of the data set in the train data matrix
    for (int i=0; i<900; i=i+1) {
        train.at(0).at(i) = pclass.at(i);
        train.at(1).at(i) = survived.at(i);
    }
    
    //filling in remaining observations of the data set in the test data matrix
    int j = 0;
    for (int i=900; i<=pclass.size()-1; i++) {
        test.at(0).at(j) = pclass.at(i);
        test.at(1).at(j) = survived.at(i);
        j++;
    }
    
    //creating data_matrix. All the rows of the first column = 1 as this column will be multiplied with the intercept later (w0). The rows of the second column will be filled with the predictor values of from train (pclass)
    vector<vector<double>> data_matrix(2, vector<double> (train.at(0).size()));
    fill(data_matrix.at(0).begin(), data_matrix.at(0).end(),1);
    data_matrix.at(1) = train.at(0);
    
    //labels will have the target values from train (survived)
    vector<double> labels = train.at(1);
    
    double learning_rate = 0.001;
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    //this loop finds the coefficients of our model.
    for (int i=0; i<500000; i++){
        vector<double> probs = sigmoid(multiplication(data_matrix, weights));
        
   
        vector <double> error = vectorSubtraction(train.at(1), probs);
        

        vector <double> temp = transposeMultiplication(data_matrix, error) ;
        for (int i= 0; i<temp.size(); i++){
            temp.at(i) = temp.at(i)*learning_rate;
        }
        
        weights = vectorAddition(weights, temp);
    }
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    //creating test_matrix. All the rows of the first column = 1 as this column will be multiplied with the intercept later (w0). The rows of the second column will be filled with the predictor values from test (pclass)
    vector<vector<double>> test_matrix(2, vector<double> (test.at(0).size()));
    fill(test_matrix.at(0).begin(), test_matrix.at(0).end(),1);
    test_matrix.at(1) = test.at(0);
    
    //multiplies the test_matrix with the model's coefficients to get our predictions
    vector<double> predicted = multiplication(test_matrix, weights);
    

    //converts predictions to probabilities
    vector<double> probs = probabilityConversion(predicted);
    
    //converts probability vectors into 2 classes where the class is 0 if probability is < 0.5 and 1 if >= 0.5
    vector<double> predictions(probs.size());
    for (int i=0; i<probs.size(); i++){
        if (probs[i] > 0.5)
            predictions[i] = 1;
        else
            predictions[i] = 0;
    }
    
    // metrics = {TP,TN,FP,FN};
    vector<double> metrics = findMetrics(predictions, test[1]);
    
    //calculating metrics
    double accuracy = (metrics[0] + metrics[1])/(metrics[0] +metrics[1] + metrics[2] + metrics[3]);
    double  specificity =metrics[0] /(metrics[0] + metrics[3]);
    double sensitivity = metrics[1] /(metrics[1] + metrics[2]);
    
    //outputting coefficients and metrics
    cout << endl << "Coefficients:" << endl;
    cout << "Intercept: " << weights[0] << endl;
    cout << "PClass: " << weights[1] << endl;
    cout << endl << "Metrics:" << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;
    
    cout<< endl << "Runtime: " << time_span.count() << " seconds.";
    
    
    return 0;
}
