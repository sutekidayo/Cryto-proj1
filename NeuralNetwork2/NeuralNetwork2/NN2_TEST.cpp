#include "NN2.h"






static float Error(vector<float> target, vector<float> output) // sum absolute error. could put into NeuralNetwork class.
    {
      float sum = 0.0;
      for (int i = 0; i < target.size(); ++i)
        sum += std::abs(target[i] - output[i]);
      return sum;
    }

// returns a random float between min and max
inline float randDouble(const float min, const float max)
{ return rand()/(float(RAND_MAX)+1)*(max-min)+min; }

vector<float> stringToInput(int numInputs, string input)
{
	vector<float> generatedInput(numInputs);
    for (int i = 0; i < numInputs; i++)
		generatedInput[i] = (float)input[i];  // simple type cast to floats
	return generatedInput;
}




int main()
{
	NN2 NN(64,128,1);

	float eta = .90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
    float alpha = 0.14; // momentum - to discourage oscillation. found by trial and error.
	vector<float> t_values(1);
	t_values[0] = .9999;// t_values[1] = 0.9; t_values[2] = .6;

	vector<float> randWeights(NN.getNumWeights());
	for (int i = 0; i < randWeights.size(); i++)
		randWeights[i] = randDouble(-1.0,1.0);
	NN.SetWeights(randWeights);
	string English = "This is sample English Text. Yup, English.  You are going to be trainined to recognize english language. Even if it is poorly typed, or their are typos.  Like twitter...";
	string notEnglish = "-2358-29fj-9jf -jf -wijf-sizdjf-v -8 =-2j3r['l;k'asPDof -0w9jf[slmfd':Kfdgv [pijwef ;mfd [03j[';slMDF;ODIFJG0- 8HWP4E3RTN;KEFMD";
	vector<float> xValues = stringToInput(64,English);
	int ctr = 0;
	float prevError;
    vector<float> yValues = NN.ComputeOutputs(xValues); // prime the back-propagation loop
        float error = Error(t_values, yValues);
        while (ctr < 1000 && error > 0.01)
		{
		  NN.UpdateWeights(t_values, eta, alpha);
          
          yValues = NN.ComputeOutputs(xValues);
          prevError = error;
          error = Error(t_values, yValues);

		  if (prevError != error)
			cout << "Error = " << error << std::endl;
          ++ctr;
		}
       NN.GetWeightsToFile("english_recog.nn");


	   string stringtoTest;

	   cout << "Type some english!\n";
	   std::getline(std::cin,stringtoTest);

	   xValues = stringToInput(64,stringtoTest);
	   yValues = NN.ComputeOutputs(xValues);

	   cout << yValues[0] << "% chance that its english";



	return 0;
}

