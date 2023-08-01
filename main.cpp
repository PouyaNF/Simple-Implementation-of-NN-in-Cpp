//
// Created by pooya on 5/22/2023.
//
#include <vector>
#include <iostream>
#include <cassert>
#include "Net.h"
#include "TrainingData.h"


using namespace std;



void showVectorVals(const string &label, vector<double> &v){
    cout << label << " ";
    for (double i : v) cout << i << " ";
    cout << endl;
}

void train (const string& trainingFilename){

    TrainingData trainData(trainingFilename);
    vector<unsigned> topology;
    trainData.getTopology(topology);
    Net neuralNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0 ;
    while (!trainData.isEof() ){
        ++trainingPass ;
        cout<< endl << "sample " << trainingPass << endl;

        // get new input data and feed it forward
        if ( trainData.getNextInputs(inputVals) != topology[0] ) break;
        showVectorVals("Inputs:", inputVals);
        neuralNet.feedForward(inputVals);

        // collect new results
        neuralNet.getResults(resultVals);
        showVectorVals("Outputs:",  resultVals);

        // compare the results with targets Vals and backpropagation
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
        neuralNet.backProp(targetVals);

        cout << "recent average error: " <<  neuralNet.getRecentAverageError() << endl;
    }
    cout << endl<< "Done" << endl;
}


int main(){

    string filename = "trainingData.txt";
    string abs_filename = "trainingData.txt";
    train(abs_filename);
    return 0;
}


