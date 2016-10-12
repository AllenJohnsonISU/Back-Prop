//This helps compiler allocate enough memory
#pragma once //makes it so the main will only load once, even if used multiple times
#include <vector>
#include <functional>
#include <fstream>

class Perceptron
{
public:
    Perceptron(unsigned int); //Constructor, has to be same name as class
    //Perceptron(std::fstream); used to load a file of inputs
    //function (standard to have lower case first letter)
    
    Perceptron(std::vector<double>, double, double); // Added for Neural Network, add to cpp too
    
    //void setActivationFunction(const std::function<int(double)>&);
    void setActivationFunction(const std::function<double(double)>&); //change for NN
    
    //int computeOutput(const std::vector<double>&) const;//vector is a dynamically resizing array
    void computeOutput(const std::vector<double>&); // int to void, const removed for NN ???? (probably not)
    
    double getOutput() const; //Added for NN reduces use of computeOutput function
    
    void train(const std::vector<std::vector<double>>&, const std::vector<int>&, double, unsigned int);
    
    //Added for NN Back Propagation
    std::vector<double> getWeight() const;
    
    double getBias() const;
    double getBiasDelta() const;
    
    void updateBias(double, double);
    
private:
    //constant value that cannot change
    constexpr static const double WEIGHT_MIN = -0.01;
    constexpr static const double WEIGHT_MAX = 0.01;
    
    void update(const std::vector<double>&, int, int, double);
    double initializeRandom() const;
    
    void writeState(std::ofstream& file) const;
    
    std::vector<double> weights;
    double bias;
    double biasDelta;
    double output; //added for getOutput function
    
    //std::function<int(double)> activation;
    std::function<double(double)> activation; //Changed int to double, matchs above
    
    
};


