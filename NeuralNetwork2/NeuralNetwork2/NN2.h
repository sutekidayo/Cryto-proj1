// NN2.H
// Travis Payton
// Feb 25th 2013
// A new NeuralNetwork written for back propogation,
// and possible implementation via AMP

#ifndef NN2_H_INCLUDED
#define NN2_H_INCLUDED


#include <vector>
using std::vector;
#include <iostream>
using std::cin;
using std::cerr;
using std::cout;
#include <string>
using std::string;
#include <cmath>
using std::tanh;
using std::exp;
#include <sstream>
#include <fstream>

class NN2
{
private:
  int numInput;
  int numHidden;
  int numOutput;
  int numWeights;
  

  // Input to Hidden Layers collection of Weights
  vector<float> inputs;
  vector<vector <float>> ihWeights;
  vector<float> ihSums;
  vector<float> ihBiases;
  vector<float> ihOutputs;

  // Hidden to Output Layer collection of Weights

  vector<vector<float>> hoWeights;
  vector<float> hoSums;
  vector<float> hoBiases;
  vector<float> outputs;
  

  // Vectors for Backpropogation

  vector<float> oGrads; // output gradients
  vector<float> hGrads; // hidden gradients

  // Vectors that hold the deltas from the previous iteration
  vector<vector<float>> ihPrevWeightsDelta;
  vector<float> ihPrevBiasesDelta;
  vector<vector<float>> hoPrevWeightsDelta;
  vector<float> hoPrevBiasesDelta;
  
public:

  // Constructor, initialize the neural network
  NN2(int numI, int numH, int numO) {
     
	 numInput=numI; numHidden=numH; numOutput=numO;
	 numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	 inputs.resize(numInput);
	 ihSums.resize(numHidden);
	 ihBiases.resize(numHidden);
	 ihWeights = MakeMatrix(numInput,numHidden);
	 ihOutputs.resize(numHidden);

	 hoWeights = MakeMatrix(numHidden, numOutput);
	 hoSums.resize(numOutput);
	 hoBiases.resize(numOutput);
	 outputs.resize(numOutput);


	 oGrads.resize(numOutput);
	 hGrads.resize(numHidden);

	 ihPrevWeightsDelta = MakeMatrix(numInput,numHidden);
	 ihPrevBiasesDelta.resize(numHidden);
	 hoPrevWeightsDelta = MakeMatrix(numHidden,numOutput);
	 hoPrevBiasesDelta.resize(numOutput);
 
  }

  int getNumWeights()
  {
     return numWeights;
  }

  void UpdateWeights(vector<float> tValues, float eta, float alpha) {
  // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)

      // 1. compute output gradients
      for (int i = 0; i < oGrads.size(); i++)
      {
        float derivative = (1 - outputs[i]) * (1 + outputs[i]); // derivative of tanh
        oGrads[i] = derivative * (tValues[i] - outputs[i]);
      }

      // 2. compute hidden gradients
      for (int i = 0; i < hGrads.size(); i++)
      {
        float derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
        float sum = 0.0;
        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
          sum += oGrads[j] * hoWeights[i][j]; // each downstream gradient * outgoing weight
        hGrads[i] = derivative * sum;
      }

      // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
      for (int i = 0; i < ihWeights.size(); i++) // 0..2 (3)
      {
        for (int j = 0; j < ihWeights[0].size(); ++j) // 0..3 (4)
        {
          float delta = eta * hGrads[j] * inputs[i]; // compute the new delta
          ihWeights[i][j] += delta; // update
          ihWeights[i][j] += alpha * ihPrevWeightsDelta[i][j]; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
          ihPrevWeightsDelta[i][j] = delta;
		}
      }

      // 3b. update input to hidden biases
      for (int i = 0; i < ihBiases.size(); i++)
      {
        float delta = eta * hGrads[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
        ihBiases[i] += delta;
        ihBiases[i] += alpha * ihPrevBiasesDelta[i];
		ihPrevBiasesDelta[i] = delta;
      }

      // 4. update hidden to output weights
      for (int i = 0; i < hoWeights.size(); i++)  // 0..3 (4)
      {
        for (int j = 0; j < hoWeights[0].size(); ++j) // 0..1 (2)
        {
          float delta = eta * oGrads[j] * ihOutputs[i];  // see above: ihOutputs are inputs to next layer
          hoWeights[i][j] += delta;
          hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
          hoPrevWeightsDelta[i][j] = delta;
        }
      }

      // 4b. update hidden to output biases
      for (int i = 0; i < hoBiases.size(); i++)
      {
        float delta = eta * oGrads[i] * 1.0;
        hoBiases[i] += delta;
        hoBiases[i] += alpha * hoPrevBiasesDelta[i];
        hoPrevBiasesDelta[i] = delta;
      }
    } // UpdateWeights

  void SetWeights(vector<float> weights) {
  // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
     
      int k = 0; // points into weights param

      for (int i = 0; i < numInput; i++)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];

      for (int i = 0; i < numHidden; i++)
        ihBiases[i] = weights[k++];

      for (int i = 0; i < numHidden; i++)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];

      for (int i = 0; i < numOutput; i++)
        hoBiases[i] = weights[k++];
  }

  bool SetWeightsFromFile(string inputfile ) {
	//load the file
	std::ifstream file(inputfile.c_str());

	// file does not exist, or didn't open
	if (!file) {
		std::cerr<<"ERROR: Cannot read from file \n";
        return false;}
    // go until the end of the file is reached
	vector <float> tmpweights;
    while(!file.eof()) {
		std::string line;
        unsigned int float_bits;

        // if read error, not due to EOF
		if (!file && !file.eof()) return false;

		while(getline(file, line, ','))
		{
			//more crazy pointer magic to convert the bits to a float
			file >> float_bits;
			unsigned int * float_bits_addr = &float_bits;
			float score = *(float *)float_bits_addr;
			tmpweights.push_back(score);
		}

    }
	SetWeights(tmpweights);
	return true; //read completed without errors
}
  
  vector<float> GetWeights() {
 
      vector<float> result(numWeights);
      int k = 0;
      for (int i = 0; i < ihWeights.size(); i++)
        for (int j = 0; j < ihWeights[0].size(); ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < ihBiases.size(); i++)
        result[k++] = ihBiases[i];
      for (int i = 0; i < hoWeights.size(); i++)
        for (int j = 0; j < hoWeights[0].size(); ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < hoBiases.size(); i++)
        result[k++] = hoBiases[i];
      return result;
  }

// GetWeightsToFile
// This function takes gets the weights and out puts into a CSV file
// Pre Conditions:  NONE
// Post Conditions:  The data is stored in a file of the specified name

void GetWeightsToFile (string filename)
{
	std::ofstream csvfile;
	csvfile.open(filename.c_str());
	vector <float> weights = GetWeights();
	for (int i = 0; i < weights.size(); i++)
	{
		   //and now for crazy pointer magic to store the bits of a float reather than risk losing precision
			const float * ptr = &(weights[i]);
			csvfile<<(*(unsigned int *)ptr) << " , ";
	}
	csvfile.close();
}

  vector<float> ComputeOutputs(vector<float> xValues) {


      for (int i = 0; i < numHidden; i++)
        ihSums[i] = 0.0;
      for (int i = 0; i < numOutput; i++)
        hoSums[i] = 0.0;

      inputs = xValues; // copy x-values to inputs
      
      for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden weighted sums
        for (int i = 0; i < numInput; i++)
          ihSums[j] += inputs[i] * ihWeights[i][j];

      for (int i = 0; i < numHidden; i++)  // add biases to input-to-hidden sums
        ihSums[i] += ihBiases[i];

      for (int i = 0; i < numHidden; i++)   // determine input-to-hidden output
        ihOutputs[i] = SigmoidFunction(ihSums[i]);

      for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
        for (int i = 0; i < numHidden; i++)
          hoSums[j] += ihOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; i++)  // add biases to input-to-hidden sums
        hoSums[i] += hoBiases[i];

      for (int i = 0; i < numOutput; i++)   // determine hidden-to-output result
        outputs[i] = HyperTanFunction(hoSums[i]);

      vector<float> result = outputs; // could define a GetOutputs method instead
      

      return result;
    } // ComputeOutputs
  

private:

  vector<vector<float>> MakeMatrix(int rows, int cols)
  {
  vector<vector<float>> result(rows,vector<float>(cols));
  return result;
  }
  
  float SigmoidFunction(float x)
  {
    if (x < -45.0) return 0.0;
    else if (x > 45.0) return 1.0;
    else return 1.0 / (1.0 + exp(-x));
  }

  float HyperTanFunction(float x)
  {
    if (x < -10.0) return -1.0;
    else if (x > 10.0) return 1.0;
    else return tanh(x);
  }
}; // End NN2 Class




#endif