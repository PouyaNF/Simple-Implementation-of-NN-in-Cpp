//
// Created by pooya on 5/23/2023.
//

#include "Neuron.h"
#include <cmath>
using namespace std;


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

    for (unsigned c =0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

double Neuron::randomWeight() {

    // return double num between 0 and 1
    return rand() / double (RAND_MAX);
}

void Neuron::setOutputVal(double val) { m_outputVal = val; }

double Neuron::getOutputVal() const {return m_outputVal;}

double Neuron::activationFunc(double x)
{
    // tanh - out put range [-1 , 1]
    return tanh(x);
}

double Neuron::activationFuncDerivative(double x)
{
    // approximation of the derivative of tanh - it is faster
    return 1.0 - (x*x);
}


void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::activationFunc(sum);
}


void Neuron::updateInputWeights(Layer &prevLayer)
{

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight = eta * neuron.getOutputVal()*m_gradient + alpha*oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;

        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}


double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFuncDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationFuncDerivative(m_outputVal);
}