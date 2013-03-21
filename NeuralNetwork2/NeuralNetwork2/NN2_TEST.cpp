#include "NN2.h"
#define TEXT_TRAINING
//#define TEXT_TESTING  // uncomment this line to test the Neural Network
//#define IMAGE_TESTING
//#define IMAGE_TRAINING
#include <fstream>
using std::ifstream;
#include <ctime>
#include "ppm.h"

static float Error(vector<float> target, vector<float> output) // sum absolute error. could put into NeuralNetwork class.
{
	float sum = 0.0;
	for (int i = 0; i < target.size(); ++i)
		sum += std::abs(target[i] - output[i]);
	return sum;
}

// returns a random float between min and max
inline float randFloat(const float min, const float max)
{ return rand()/(float(RAND_MAX)+1)*(max-min)+min; }

vector<float> stringToInput(string input)
{
	// Inputs of Neural Network
	// Now based on frequency of that input in the sample text  -- so each input is divided by total input length
	// Inputs 0  - 25 == a - z or A - Z  characters   
	// Inputs 26      == 0 - 9  digit
	// Inputs 27      == { } @ # $ % ^ & * ( ) < > ? / \ : ; ' "   total number of symbols 
	// Inputs 28      ==  . , ! ?  sentence endings 
	// Inputs 29      == th   
	// Inputs 30      == ing
	// Inputs 31      == ed 
	// Inputs 32      == of
	// Inputs 33      == an
	// inputs 34      == nd
	// inputs 35      == space
	vector<float> generatedInput(36);
	std::fill(generatedInput.begin(),generatedInput.end(),0);
	for (int i = 0; i < input.size(); i++){
		if (isalpha(input[i]))
		{
			if(input[i] >= 65 && input[i] <= 90){
				input[i]+=32; // convert to lowercase
			}
			// match 'th'
			if(input[i] == 't')
				if(input[i+1] == 'h')
					generatedInput[29] += 1.0/input.size();
			// match 'ing'
				else if(input[i] == 'i')
					if(input[i+1] == 'n')
						if(input[i+2] == 'g')
							generatedInput[30] += 1.0/input.size();
			// match 'ed'
						else if(input[i] == 'e')
							if(input[i+1] == 'd')
								generatedInput[31] += 1.0/input.size();

			//match 'of'
							else if(input[i] == 'o')
								if(input[i+1] == 'f')
									generatedInput[32] += 1.0/input.size();
			//match 'an'
								else if(input[i] == 'a')
									if(input[i+1] == 'h')
										generatedInput[33] += 1.0/input.size();
			//match 'nd'
									else if(input[i] == 'n')
										if(input[i+1] == 'd')
											generatedInput[34] += 1.0/input.size();

			// increment number of letters
			generatedInput[(int)(input[i] - 97)] += 1.0/input.size();
		}
		else if (isdigit(input[i]))
		{
			generatedInput[26] += 1.0 / input.size();
		}
		else if ((input[i] >= 34 && input[i] <= 47) || (input[i] >= 58 && input[i] <= 62) ||
			(input[i] >= 91 && input[i] <= 96) || (input[i] >= 123 && input[i] <= 126) ||
			(input[i] == '@'))
			generatedInput[27] += 1.0/input.size();


		else if (input[i] == '.' || input[i] == ',' || input[i] == '!' || input[i] == '?')
			generatedInput[28] += 1.0/ input.size();

		else if (input[i] == ' ')
			generatedInput[35] += 1.0/input.size();
	}

	return generatedInput;
}

bool trainNN(vector<string> content, NN2 &NN)
{
	vector<float> xValues;
	vector<float> t_values(1);
	vector<float> yValues;
	vector<float> randXValues(64);
	float error2 = 0;
	int ctr;
	for (int i = 0;i< content.size(); i++)
	{ //Load the first file
		ifstream file(content[i].c_str(),ifstream::in);
		if(!file)
		{
			cerr<<"Error Opening File! \n";
			continue;
		}
		vector<float> input;
		cout << "Reading file " << i << std::endl;
		cout << "Error with Random Text                             Error with English\n";
		while(!file.eof()) {
			string line;

			// if read error, not due to EOF
			if (!file && !file.eof()) return false;
			int cnt = 0;
			while(std::getline(file,line))
			{
				if ( line.find_first_not_of(" \t\n\v\f\r") != std::string::npos)
				{

					xValues = stringToInput(line);

					float prevError = 0;
					float eta = .0350;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
					float alpha = 0.20; // momentum - to discourage oscillation. found by trial and error.
					t_values[0] = .99;
					yValues = NN.ComputeOutputs(xValues); // prime the back-propagation loop
					//cout << line << "\nEvaluated as " << yValues[0];
					float error = Error(t_values, yValues);

					while (ctr < 500 && error > .001)
					{
						NN.UpdateWeights(t_values, eta, alpha);

						yValues = NN.ComputeOutputs(xValues);
						prevError = error;
						error = Error(t_values, yValues);

						//if (prevError != error)
						//cout << "Error: " << error << "\n";
						ctr++;
					}
					ctr=0;


					cnt++;
					if (cnt%100 == 0)
						cout << "      "<< error2 << "                                  "<<error <<"\n";


				}

			}
			//cout <<"Training on Bad Output \n";
			// train on bad input too
			t_values[0] = .00001;
			for (int loop=0; loop < 5000; loop++){
				for (int i=0; i<64;i++)
					randXValues[i]=randFloat(.0, 1);
				yValues = NN.ComputeOutputs(randXValues); // prime the back-propagation loop
				error2 = Error(t_values, yValues);
				float eta = .035;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
				float alpha = 0.15;
				while (ctr < 500 && error2 > .001)
				{
					NN.UpdateWeights(t_values, eta, alpha);
					yValues = NN.ComputeOutputs(randXValues);
					//cout <<"Got new yValue " << yValues[0] << "\n";

					error2 = Error(t_values, yValues);
					//if (prevError != error2)
					//cout << "Error2: " << error2 << "\n";


					ctr++;
				}
				ctr = 0;

			}
		}
		NN.GetWeightsToFile("english_recog.nn");
	}
}



int main()
{






#ifdef TEXT_TESTING
	NN2 NN(36,128,1);
	if(!NN.SetWeightsFromFile("english_recog.nn"))
	{
		cout << "Error loading weights!!";
		return 0;
	}

	string test;
	while (true)
	{
		cout << "Enter text to evaluate..\n";
		std::getline(std::cin, test);
		vector<float> xValues = stringToInput(test);
		vector<float> yValues = NN.ComputeOutputs(xValues);
		cout << "Chance that it is english: " << yValues[0] * 100 << "%\n";


	}
#endif //TEXT_TESTING

#ifdef IMAGE_TRAINING
	//Image trainimg("trainingimg.ppm");


	/*NN2 NN(trainimg.w*trainimg.h,trainimg.w,1);
	vector<float> randweights(NN.getNumWeights());
	srand(time(NULL));
	for (int i=0;i<randweights.size();i++)
	randweights[i]=randFloat(-4.0,4.0);

	NN.SetWeights(randweights);



	vector <float> x_values(trainimg.w*trainimg.h);
	// have to get the pixel values into the input
	for (int y=0; y < trainimg.h; y++)
	for (int x=0; x < trainimg.w; x++){
	x_values[y*trainimg.w+x] = trainimg.at(x,y).v[0]+trainimg.at(x,y).v[1]+trainimg.at(x,y).v[2];
	}

	cout << "starting to Train.." << std::endl;
	int ctr = 0;
	float prevError;

	float eta = .90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
	float alpha = 0.41; // momentum - to discourage oscillation. found by trial and error.
	vector<float> t_values(1);
	t_values[0] = .9999; // t_values[1] = 0.9; t_values[2] = .6;
	vector<float> yValues = NN.ComputeOutputs(x_values); // prime the back-propagation loop

	float error = Error(t_values, yValues);

	while (ctr < 1000 || error > .01)
	{
	NN.UpdateWeights(t_values, eta, alpha);

	yValues = NN.ComputeOutputs(x_values);
	prevError = error;
	error = Error(t_values, yValues);

	if (prevError != error)
	cout << "Error: " << error << "\n";
	++ctr;
	}
	NN.GetWeightsToFile("img_recog.nn");
	*/
#endif

#ifdef IMAGE_TESTING
	Image trainimg("test.ppm");
	NN2 NN(200*133,200,1);
	NN.SetWeightsFromFile("img_recog.nn");
	vector <float> x_values(trainimg.w*trainimg.h);
	// have to get the pixel values into the input
	for (int y=0; y < trainimg.h; y++)
		for (int x=0; x < trainimg.w; x++){
			x_values[y*trainimg.w+x] = trainimg.at(x,y).v[0]+trainimg.at(x,y).v[1]+trainimg.at(x,y).v[2];
		}
		vector<float> yValues = NN.ComputeOutputs(x_values);
		cout << "Probability that it is the training image = " << yValues[0] * 100 << "%\n";

#endif

#ifdef TEXT_TRAINING 

		NN2 NN(36,128,1);
		srand(time(NULL));
		vector<float> randWeights(NN.getNumWeights());

		for (int i=0; i < randWeights.size(); i++)
			randWeights[i] = randFloat(-1.0,1.0);

		NN.SetWeights(randWeights);

		vector<string> trainings;
		trainings.push_back("1.txt");
		trainings.push_back("2.txt");
		trainings.push_back("3.txt");
		trainings.push_back("4.txt");
		trainings.push_back("5.txt");
		trainings.push_back("6.txt");
		trainings.push_back("7.txt");
		trainings.push_back("8.txt");
		trainings.push_back("9.txt");
		trainings.push_back("10.txt");
		trainings.push_back("11.txt");
		trainings.push_back("12.txt");
		trainings.push_back("13.txt");
		trainings.push_back("14.txt");
		trainings.push_back("15.txt");



		trainNN(trainings, NN);
#endif





		return 0;
}
