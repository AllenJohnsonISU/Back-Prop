#include "NeuralNetwork.hpp"
#include "Perceptron.hpp"
#include <cmath>

#include <iostream> //for outputing tests

using namespace std;

NeuralNetwork::NeuralNetwork(vector<int> layerNodes) {
    //takes input of layer nodes (i,j,k) and creates brain
    this->layerNodes = layerNodes;
    
    initializeLayers();
}

// Function to initialize the layers in NN
void NeuralNetwork::initializeLayers() {
    //sample input (3,4,2)
    for(unsigned int i = 0; i <layerNodes.size(); i++) {
        int numNodes = layerNodes[i]; //gets number of nodes for each layer
        
        vector<Perceptron> layerPerceptrons; //creates vector of perceptrons for each layer
        
        for(unsigned int j = 0; j < numNodes; j++) {
            
            if (i == 0) { //Creates input layer
                Perceptron node({1.0}, 0.0, 0.0); //sets (weights, bias, bias delta)
                
            //input must be output
                node.setActivationFunction([](double value) { //ask to explain lamda function better
                    return value;
                });
                
                layerPerceptrons.push_back(node); //sets value
                
            } else if (i == layerNodes.size()-1) { //Creates output layer
                Perceptron node(layerNodes[i-1]);
                
                //creates output layer activation functions
                node.setActivationFunction([](double value) { //ask to explain lamda function better
                    return value;
                });
                
                layerPerceptrons.push_back(node);
                
            } else { //Creates hidden layer (middle)
                Perceptron node(layerNodes[i-1]); // [i-1] b/c 0 indexed
                //Set hidden layer activation functions to tanh
                node.setActivationFunction(NeuralNetwork::hyperTan);
                
// Changed to have hyperTan is own function
//      [](double value) { //ask to explain lamda function better
//                    if (value < -20) {
//                        return -1.;
//                    } else if (value > 20) {
//                        return 1.
//                    } else {
//                        return tanh(value);
//                    }
//                                               
//                });
                
                layerPerceptrons.push_back(node);
            }
            
        }
        brain.push_back(layerPerceptrons);
    }
}

vector<double> NeuralNetwork::predict(vector<double> _inputs) {
    /*
     1) Feed inputs into input layer
     2) Compute outputs of each layer
     3) Apply softmax on output layer
    */
    
    if (layerNodes[0] != _inputs.size()) {
        throw "Input vector size does not match input layers"; 
    }
    
    for (unsigned int i = 0; i < brain.size(); i++) {
        int numNodes = layerNodes[i];
        
        for (unsigned int j = 0; i < numNodes; i++) {
            //input layers
            if (i == 0) {
                brain[i][j].computeOutput({_inputs[j]});
                //computes each input layer by feeding it vector of input
            } else {
                brain[i][j].computeOutput(generateVector(brain[i-1]));
            }
        }
    }
    
    vector<double> output = softmax(brain[brain.size()-1]);
    return output;
};


vector<double> NeuralNetwork::generateVector(vector<Perceptron> layerPerceptrons) {
    
    vector<double> layerOutputs;
    for (unsigned int i = 0; i < layerPerceptrons.size(); i++) {
        layerOutputs.push_back(layerPerceptrons[i].getOutput()); //gets each output and pushes it into layer ouput vector
    }
    return layerOutputs;
};


vector<double> NeuralNetwork::softmax(vector<Perceptron> perceptrons) {
    
    //loop finds max value for scale factor
    double max = perceptrons[0].getOutput();
    for (unsigned int i = 1; i < perceptrons.size(); i++) {
        if (perceptrons[i].getOutput() > max) {
            max = perceptrons[i].getOutput();
        }
    }
    
    double scale = 0.;
    for (unsigned int i = 0; i < perceptrons.size(); i++) {
        scale += exp(perceptrons[i].getOutput()-max);
    }
    
    vector<double> output;
    
    for (unsigned int i = 0; i < perceptrons.size(); i++) {
        output.push_back(exp((perceptrons[i].getOutput()-max)/scale));
    }
    return output;
};


double NeuralNetwork::hyperTan(double value) {
    
    if (value < -20) {
        return -1.;
    } else if (value > 20) {
        return 1.;
    } else {
        return tanh(value);
    }
};


void NeuralNetwork::backPropagation(vector<double> _inputs, vector<double> _desired, int _epochs, double _learningrate, double _momentum) {
    
    vector<double> outputs = predict(_inputs);
    ograd = computeOutputGrad(_desired, outputs);
    hgrad = computeHiddenGrad(ograd);
    computeHiddenBias(hgrad, _learningrate, _momentum);
    
    //set new weights for each perceptron node
    
    
};


vector<double> NeuralNetwork::computeOutputGrad(vector<double> _desired, vector<double> outputs) {
    
    vector<double> ograd;
    for (unsigned int i = 0; i < outputs.size(); i++) {
        ograd.push_back((1-outputs[i])*(outputs[i])*(_desired[i]-outputs[i]));
    };
    return ograd;
};


vector<vector<double>> NeuralNetwork::computeHiddenGrad(vector<double> ograd) {
    
    vector<vector<double>> hgrad;
    hgrad.resize(3, vector<double> (1,0.)); //change to be dynamic
    double derivative;
    double sum = 0.0;
    
    for (unsigned int i = 1; i < layerNodes.size()-1; i++) {
        for (unsigned int j = 0; j < layerNodes[i]; j++) {
            vector<double> weights = brain[i][j].getWeight();
            
            derivative = (1-brain[i][j].getOutput())*(1+brain[i][j].getOutput());
            for (unsigned int k = 0; k < ograd.size(); k++) {
                sum += ograd[k]*weights[0];
            }
            hgrad[i][j] = derivative*sum;
        }
    }

    return hgrad;
};


void NeuralNetwork::computeHiddenBias(vector<vector<double>> hgrad, int _learningrate, int _momentum) {
    double mfactor;
    double delta;
    double bias;
    
    for (unsigned int i = 1; i < layerNodes.size(); i++) {
        for (unsigned int j = 0; j < layerNodes[i]; j++) {

            mfactor = brain[i][j].getBiasDelta()*_momentum;
            delta = hgrad[i][j]*_learningrate;
            bias = brain[i][j].getBias()+delta+mfactor;
            brain[i][j].updateBias(bias, delta); //updates bias and bias delta
        }
    }
    
};




void NeuralNetwork::computeHiddenWeights(vector<double> hgrad, double _learningrate, double _momentum) {
    vector<double> fetweights;
    double delta;
    double weight;
    
    brain[i][j].getWeights();
    delta  = _learningrate*hgrad;
    weight = (fetweights+delta);
};



