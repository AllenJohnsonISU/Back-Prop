#include "Perceptron.hpp"
#include <algorithm> //iterator (see train)
#include <cstdlib>
#include <iostream> // add for NN (appears to allow cout)

using namespace std;

Perceptron::Perceptron(unsigned int _inputSize) {
    for(unsigned int i = 0; i < _inputSize; ++i) {
        weights.push_back(initializeRandom());
    }
    
    bias = initializeRandom();
    output = 0.; //added for getOutput function
}


// Added new constructor for NN
Perceptron::Perceptron(std::vector<double> _weights, double _bias, double _biasDelta) {
    for(unsigned int i = 0; i < _weights.size(); ++i) {
        weights.push_back(_weights[i]);
        
    }
    
    bias = _bias;
    biasDelta = _biasDelta;
    output = 0.; //added for getOutput function
}


void Perceptron::setActivationFunction(const std::function<double(double)> & _activationFunction) {
    activation = _activationFunction;
}


//Added getOutput function to reduce repeated computations
double Perceptron::getOutput() const {
    return output;
}


//int Perceptron::computeOutput(const vector<double>& _inputs) {
void Perceptron::computeOutput(const vector<double>& _inputs) { //int to void, store info as output variable
    
    if (_inputs.size() != weights.size() || !activation) {
        //return 0;
        throw 10;
        //cout << "Weights and Inputs do not match" << endl; //throws exception
    }
   
    double sum = 0.;
    for(unsigned int i = 0; i < weights.size(); ++i) {
        sum += weights[i] * _inputs[i];
    }
    sum += bias;
    //return activation(sum);
    output = activation(sum); //added to reduce repeated computation
};


void Perceptron::train(const vector<vector<double>>& _inputSet, const vector<int>& _outpuSet, double _alpha, unsigned int _maxEpoch) {
    
    ofstream outputFile("NNPerceptronState.csv");
    
    outputFile << "Interation";
    for(unsigned int i = 0; i < weights.size(); ++i) {
        outputFile << ", weight" << i;
    }
    outputFile << ", bias\n";
    
    
    vector<unsigned int> indicies;
    for(unsigned int i = 0; i <weights.size(); ++i) {
        indicies.push_back(i);
    }
    
    for (unsigned int epochs = 0; epochs < _maxEpoch; ++epochs) {
        random_shuffle(indicies.begin(), indicies.end());//random_shuffle is part of algorithm
    
        for(unsigned int i = 0; i < _inputSet.size(); ++i) {
            int desired = _outpuSet[i];
            //int computed = computeOutput(_inputSet[i]); // Changed for NN additions
            computeOutput(_inputSet[i]);// stores to output
            int computed = getOutput();//fetches the computed value
            
            update(_inputSet[i], desired, computed, _alpha);
        }
        
        outputFile << epochs << ", ";
        writeState(outputFile);
    }
}


void Perceptron::update(const vector<double>& _inputs, int _desired,
                        int _computed, double _alpha) {
    
    if(_desired == _computed)
        return;
    
    int delta = _computed - _desired;
    double factor = _alpha * delta;
    
    for(unsigned int i = 0; i < _inputs.size(); ++i) {
        if(_inputs[i] >= 0.) {
            weights[i] -= factor * _inputs[i];
        }
        else {
            weights[i] += factor * _inputs[i];
        }
    }
    
    bias -= factor;
}


void Perceptron::updateBias(double _bias, double _biasDelta) {
    bias = _bias;
    biasDelta = _biasDelta;
}

//void Perceptron::updateWeight( _weights) {
//    weights = _weights;
//}

double Perceptron::initializeRandom() const{
    return ((rand() % 10001) / 10000.) * (WEIGHT_MAX - WEIGHT_MIN)
    + WEIGHT_MIN;
}


void Perceptron::writeState(ofstream& file) const {
    for(unsigned int i = 0; i < weights.size() ; ++i)
        file << weights[i] << ", ";
    
    file << bias << '\n';
}


//Add for NN Back-Progration
vector<double> Perceptron::getWeight() const{
    return weights;
}

double Perceptron::getBias() const{
    return bias;
}

double Perceptron::getBiasDelta() const{
    return biasDelta;
}

