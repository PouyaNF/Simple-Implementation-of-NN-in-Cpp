//
// Created by pooya on 6/2/2023.
//
#include "TrainingData.h"
#include <sstream>
#include <iostream>

TrainingData::TrainingData(const string& filename) {
    // filename.c_str() converts the string object filename to a C-style string
    // (a const char*) which is expected by the open() function.
    m_trainingDataFile.open(filename.c_str());
    if (!m_trainingDataFile.is_open()) {
        cout << "Failed to open the file." << endl;
    }
}


void TrainingData::getTopology(vector<unsigned int> &topology) {
    string line;
    string label;
    getline (m_trainingDataFile , line);
    stringstream  ss(line);  // This line creates a string stream object named ss
    // and initializes it with the contents of the line string.
    ss >> label; // the first element from ss is extracted and
    // stored in the label string variable.
    if (this->isEof() ||  label.compare("topology:") !=0  )
    {
        abort();
    }
    while (!ss.eof()){
        unsigned n;
        ss >>n ;
        topology.push_back(n);
    }
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals) {
    inputVals.clear();
    string line;
    string label;
    getline(m_trainingDataFile , line );
    stringstream ss(line);
    ss >> label;
    // if the line in label start with 'in: '
    if (label == "in:"){
        double oneVal;
        while (ss >> oneVal) {
            inputVals.push_back(oneVal);
        }
    }
    return inputVals.size();
}


unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals) {
    targetOutputVals.clear();
    string line;
    string label;
    getline(m_trainingDataFile , line );
    stringstream ss(line);
    ss >> label;
    // if the line in label start with 'in: '
    if (label == "out:"){
        double oneVal;
        while (ss >> oneVal) {
            targetOutputVals.push_back(oneVal);
        }
    }
    return targetOutputVals.size();
}



// to be used to creat trainingData.txt file
void xorData() {
    string file = "C:\\Users\\pooya\\Desktop\\C++ projects\\5- implementation of NN in c++\\"
                  "simple implementation\\trainingData.txt";

    ofstream outputFile(file);  // Open the output file

    if (!outputFile.is_open()) {
        cout << "Failed to open the file." << endl;
        return;
    }

    cout << "topology: 2 4 1" << endl;
    for (int i = 2000; i >= 0; --i) {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX)); // 0 or  1
        int n2 = (int)(2.0 * rand() / double(RAND_MAX));
        int t = n1 ^ n2;  //  performs a bitwise XOR operation
        cout << "in: " << n1 << ".0 " << n2 << ".0" << endl;
        cout << "out: " << t << ".0" << endl;

        outputFile << "in: " << n1 << ".0 " << n2 << ".0" << endl;  // Write to the file
        outputFile << "out: " << t << ".0" << endl;
    }

    outputFile.close();  // Close the output file
}
