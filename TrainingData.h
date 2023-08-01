//
// Created by pooya on 6/2/2023.
//

#ifndef SIMPLE_IMPLEMENTATION_TRAININGDATA_H
#define SIMPLE_IMPLEMENTATION_TRAININGDATA_H

#include <vector>
#include <fstream>

using namespace std;


class TrainingData {

public:
    explicit TrainingData(const string& filename);
    bool isEof() {return m_trainingDataFile.eof();}
    void getTopology(vector<unsigned> &topology);
    // Returns the num of input values read from the file
    unsigned getNextInputs(vector<double > &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals) ;
private:
    ifstream m_trainingDataFile;
};


#endif //SIMPLE_IMPLEMENTATION_TRAININGDATA_H
