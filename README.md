# autoencoder
a simple keras autoencoder made by chatgpt

pass0.ipynb : setting epsilon = 0 the data array still differs from the perturbed data

foo_2a.py : this code uses https://blog.keras.io/building-autoencoders-in-keras.html and some advice in:
https://stackoverflow.com/questions/76752336/inconsistent-output-from-an-auto-encoder

there is a difference between input and predicted data due to the approximation

the randomness upon repeating the same run several times was not resolved 

applying :
autoencoder to the input data 

or

encoder on input data followed by
decoder on latent vector 

gives the same result which asserts some correctness in the code

