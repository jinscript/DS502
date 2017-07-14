# Jin's DS502 homework

Layout:
* Zhu_Homework:
    * homework1
    * homework2
    * ...
* Chuck_Homework:
    * homework1
    * homework2
    * ...
* Jason_Homework:
    * homework1
    * homework2
    * ...

Please follow README.md in each homework's directory

Dr Zhu's Homework Architecture Overview:

common.BaseModel is an abstract class to be inherited from individual models. Internally it models each epoch of training with a forward pass and a backward pass. It provides a generic way for iterative training. Each model needs to implement some abstract method for its specialized logic.
