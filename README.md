# Generic Image Classifiers 
Hello everyone ,welcome to my Github Repo . <br/>
While developing my GAN project (for which i will also post codes) ,I also had to train some classfiers for the pipeline. <br/>
I followed some very intuitive explanations on Jason Brownlee's website [MachineLearningMastery](https://machinelearningmastery.com/) .
I tried to take one step further by adding adequate number of ResNet blocks and filters to the classifiers while tuning the hyper parameters alongwith <br/>
I also trained some more classifiers for other datasets .<br/>
I am providing the code as well as trained weights .All of which are tested on Python 3.7 and Tensorflow 1.14 <br/>
Please note a code developed on some version of Tensorflow may run but might give very unexpected behaviour on other versions of Tensorflow .<br/>
If you are using Colab just use this command before running any cell **%tensorflow_version 1.x** and then restart the Runtime .<br/>
**The Machine** for all my implementations is Nvidia GTX 1650 ,AMD Ryzen 7 8GB Ram .Cuda 10.1 .<br/>

**Emnist_resnet.py** :- I had developed this ResNet type classifier for my other big project <br/>
You should change number of filters and Resnet blocks accordingly as you use more data(shots) <br/>
This model also saves the feature extractor weights along with full classifier weights .      <br/>
I have intentionally kept this models small for any learner to copy the code and see results within minutes in Google Colab <br/>

**Emnist_MobileNetV2.ipynb** :- I tried MobileNetV2 feature extractor for same purpose and i will compare my findings </br>
This jupyer notebook is self explanatory .

**To Do :-** Train on other generic datasets like ,Cifar 10 & 100 ,Flowers . Add MobileNet v2 backbone
