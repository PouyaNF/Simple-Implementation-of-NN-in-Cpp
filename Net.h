//
// Created by pooya on 5/23/2023.
//

#ifndef SIMPLE_IMPLEMENTATION_NET_H
#define SIMPLE_IMPLEMENTATION_NET_H

#include <vector>
#include "Neuron.h"

using namespace std;

class Net {
public:
    explicit Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError() const ;

private:
    //each Net has a vector of layers and each Layer is a vector of Neurons
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;  // to calculate rms error
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

#endif //SIMPLE_IMPLEMENTATION_NET_H
