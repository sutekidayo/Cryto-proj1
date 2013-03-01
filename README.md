Neural Network 2
This is a simple Back Propogating Neural Network which I will try and train to recognize English text.
If that is successfull, the next step is to create another neural network that can recognize images.

The purpose of these is to have an AI for recognizing successfull decrypts of brute force style decryptions.  

Everything is contained in NN2.H
(The number 2 is there as this is the second Neural Network that I have implemented, the first being one that evolved using genetic algorithm instead of back propogation.)

BACKGROUND

The human brain is able to tell almost instantly that it can read a piece of text.
There is no analyzing of the text that occurs beyond the fact of simple pattern recognition.
Text recognition approaches today uses many statistical approaches to determining the language of a body of text.
N-grams, dictionary lookups, many processes that require a lot of computations in order to achieve the desirable result.
The hope with using a Neural Network is to mimic the speed and way our brain determines whether or not what we are 
looking at is something we recognize.

TRAINING

The neural network will be trained with as many sources of english as possible.
Current version has 64 input nodes, and so text is simply split into sections of 64 characters,
and then each character's int value is converted to a float.  This allows for easy processing of text.  

There are infinite number of encoding schemes. The hope is no matter which one is chosen, the Neural Network will simply 
start to converge on the desired outputs by adjusting itself to the inputs.  With enough training it shouldn't matter 
the encoding scheme, as long as it is consistent.

Currently trained:
  Ender's Game 
  Speaker for the Dead
  Xenocide
  Children of the Mind
  Ender's Shadow
  Shadow of the Hegemon
  Shadow Puppets
  Shadow of the Giant
  
