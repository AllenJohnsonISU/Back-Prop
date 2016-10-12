//
//  main.cpp
//  NeuralNetwork
//
//  Created by Allen Johnson on 10/9/16.
//  Copyright © 2016 ALJ.inc. All rights reserved.
//
#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main() {
    
    //Training Data Set
    vector<double> trainingInputs = {5.0,3.0};
    vector<double> trainingOutputs = {0.5,0.3};
    int epoch = 10;
    double learningrate = .05;
    double momentum = .01;
    
    
    vector<int> layerStructure = {2,3,2};
    vector<double> predictInputs = {1.0,10.0};
    
    //Creates input, hidden, output layer structure
    NeuralNetwork NN(layerStructure);
    
    //Blocked off section for Train with Back Prop
    NN.backPropagation(trainingInputs, trainingOutputs, epoch, learningrate, momentum);
    //                (input,output, epoch, learningrate, momentum)
    // or have backPropagation functions within NeuralNetwork?
    
    //Use input values to determine
    vector<double> output = NN.predict(predictInputs);
    
    
    //Outputs computed values
    cout << "Computed outputs" << endl;
    for (unsigned int i = 0; i < output.size(); i++) {
        cout << output[i] << endl;
    }

    return 0;
}
