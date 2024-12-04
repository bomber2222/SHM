# SHM on a simple 1 dof system (mx^^ + cx^ + kx = u): the scope is to train machine learning algorithms for the identification of the 'health' of the system. Its stiffness can deteriorate in time. Once algorithms are trained, they are used for online classification of the structure. 

1_shm.py is a code where neural networks are employed for shm of the system. The NN are trained in a supervised manner.  

2_unsupervised_classification.py is a code used for health classification of the system, using an unsupervised algorithm, i.e. k-means. The scope is to classify the system's response in different health classes. 
