#pragma once

#include <vector>
#include <functional>
#include "perceptron.hpp"

class NeuralNetwork
{
public:
    // (No.Inputs, No.Outputs, No.HiddenLayers, vector of nodes in each hidden layer)
    NeuralNetwork(const std::vector<int>);

    // A vector double of inputs to predict from
    std::vector<double> predict(std::vector<double>);
    
    // Trains the Neural Network using Back Propagation
    void backPropagation(std::vector<double>, std::vector<double>, int , double, double);
    
private:
    // Stores the num of nodes for each layer
    std::vector<int> layerNodes;
    
    // A network of all the perceptrons
    std::vector<std::vector<Perceptron>> brain;
    
    // Function to initialize the layers in NN
    void initializeLayers();
    
    // Function to calculate Hyper Tangent, static b/c not specific to each instance (only in hpp)
    static double hyperTan(double); // is this never used??
    
    // Function to calculate softmax
    std::vector<double> softmax(std::vector<Perceptron>);
    
    // Function to generate a vector of outputs from the vector of
    // input percptrons
    std::vector<double> generateVector(std::vector<Perceptron>);
    
    
    //Function used for Back Propagation Specifically
    
    //Computes Output Layer Gradients
    std::vector<double> computeOutputGrad(std::vector<double>, std::vector<double>); //add &  ?
    
    //Computes Hidden Layer Gradients
    std::vector<std::vector<double>> computeHiddenGrad(std::vector<double>); //add &  ?
    
    //Stores the output layer gradients
    std::vector<double> ograd;
    
    //Stores the hidden layer gradients
    std::vector<std::vector<double>> hgrad;
    
    //Computes Hidden Bias & Bias_Delta and updates them
    void computeHiddenBias(std::vector<std::vector<double>>, int , int);
    
    //Computes Hidden weights & weights_Delta and updates them
    void computeHiddenWeights(std::vector<double> , double, double);
    
    //Computes Output Bias & Bias_Delta and updates them
    //void computeOutputBias(std::vector<std::vector<double>>, int , int);
    
};