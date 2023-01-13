# MyJax: personal rewrite of pytorch modules... in Jax

In an attempt to learn Jax, good programming practice, and AI concepts, I am creating MyJax.  This is a personal python library where I attempt to reimplement the main modules of Pytorch from scratch up through transformer networks.  IE... linear layers, convolution layers, activation functions. dropout, batch norm, RNNs, gradient decent, etc... up through what is needed for things like GPT and Stable Diffusion. 

Why PyTorch and not Jax?  Because as of the time of this writing, I like the PyTorch API, just wanted to learn how to do things in Jax.  Thus there will necesarily be some changes, but I hope to keep those minimal. 

Simple projects I want to solve with this library:

- [ ] Makemore by Andrej Karpathy up through GPT
- [ ] Stable Diffusion by Jeremy Howard of FastAI
- [ ] Simple Jax program in their examples
- [ ] Titanic Kaggle problem
- [ ] Housing Kaggle problem
- [ ] MNIST, both traditional and fashion classification problem
- [ ] Run on single or multiple GPUs and/or CPUs
- [ ] Pass basic pylint, mypy, etc... standards
- [ ] Create test suite

The hope is, if the library becomes mature enough to solve each of the above, it's a pretty stable and I will have met my goals in the first sentence.