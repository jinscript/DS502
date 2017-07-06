# Jin's DS502 homework

Layout:
* homework:
    * week1
    * week2
    * ...
* mxnet_homework:
    * week1
    * week2
    * ...

Please follow README.md in each week's directory

Homework Architecture Overview:

common.BaseModel is an abstract class to be inherited from individual models. Internally it models each epoch of training with a forward pass and a backward pass. It provides a generic way for iterative training. Each model needs to implement some abstract method for its specialized logic.
