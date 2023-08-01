//
// Created by pooya on 5/23/2023.
//
#ifndef SIMPLE_IMPLEMENTATION_NEURON_H
#define SIMPLE_IMPLEMENTATION_NEURON_H

#include <vector>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection{
    double weight;
    double deltaWeight; };

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val);
    double getOutputVal() const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    // output val of the neuron
    double m_outputVal;
    // each neuron keeps the weights that feeds to the all neurons in the next layer
    // and the corresponding delta weights
    vector<Connection> m_outputWeights;
    // this method is responsible for random initialization of weights for each neuron
    static double randomWeight() ;
    unsigned m_myIndex;
    double m_gradient; // gradient member of each neuron
    static double activationFunc(double x);
    static double activationFuncDerivative(double x);
    double sumDOW(const Layer &nextLayer) const;
    static double eta;
    static double alpha;
};

#endif //SIMPLE_IMPLEMENTATION_NEURON_H
