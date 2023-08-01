//
// Created by pooya on 5/23/2023.
//

#include "Net.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;



Net::Net(const vector<unsigned> &topology)
{
    // adding each layer according to the topology
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer()); // add new Layer to m_layers
        // num of outputs for each neuron in each layer is topology[layerNum + 1]
        // except for the last layer that the neurons in this layer have 0 outputs
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // adding neurons for each layer
        // neuronNum <= topology[layerNum] : it goes to less or equal because of adding one
        // extra neuron for bias added to each layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            if (neuronNum == topology[layerNum])
                cout << "Bias Neuron added to the "<< layerNum << " Layer" << endl;
            else  cout << "Made a Neuron Number: "<< neuronNum + 1<< endl;
        }
       m_layers.back().back().setOutputVal(1.0);
    }
}


void Net::feedForward(const vector<double> &inputVals) {

    // input size must be equal to the number of neuron on the first layer - 1
    // we have one extra bias in m_layers[0] must be subtracted
    assert(inputVals.size() == (m_layers[0].size() ) -1 && "Wrong input size. Input size must be equal to number "
                                                           "of neurons in the first layer");

    // inserting the input values into the neurons of the input layer
    for (unsigned i = 0 ; i < inputVals.size(); ++i)
    {
        // because the class member outputVal of Neuron in private, we need setOutputVal to set it
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagation#
    // loop over each layer and each neuron after input layer and call feedForward function of neuron class

    for(unsigned layerNum = 1 ; layerNum < m_layers.size(); ++layerNum )
    {
        Layer &prevLayer  = m_layers[layerNum -1];
        for(unsigned n =0 ; n < m_layers[layerNum].size() -1 ; ++n  )
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }

    }

}



void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    // going through all neurons in output layer
    for (unsigned n=0 ; n<outputLayer.size()-1 ; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;

    }

    m_error /= double((outputLayer.size() - 1));
    m_error = sqrt(m_error);   // RMS error

    // implementing the recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor) /
            (m_recentAverageError + 1.0);

    //calculate output layer grads

    for (unsigned n = 0; n < outputLayer.size() - 1 ; ++n){
        outputLayer[n].calcOutputGradients( targetVals[n] );
    }
    // calculate grads on hidden layers
    for (unsigned layerNum = m_layers.size()-2 ; layerNum>0; --layerNum )
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum +1 ];

        for (unsigned  n = 0 ; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);

        }

    }
    //update all weights in all layers

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}
double Net::m_recentAverageSmoothingFactor = 100.0;


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}



double Net::getRecentAverageError() const {return m_recentAverageError;}


