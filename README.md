<ul>
  <ul>
    This repository contains the source files for training and testing the Vehicle Make, Model and Year Prediction Task (https://www.aiforsea.com/computer-vision). As a backend CNN framework we have used kereas with theano.
  </ul>
  <li>
    <b>train.py</b> - contains the code for fine-tuing the Inception v3 CNN taken from https://github.com/fchollet/deep-learning-models. 
      <ul>
        <li>
          The Inception v3 CNN which is trained on ImageNet dataset, have been fine tuned with VMMRdb dataset (http://vmmrdb.cecsresearch.org/), where 3040 classes are used for training. 
          According to the author of the paper (http://vmmrdb.cecsresearch.org/papers/VMMR_TSWC.pdf), they have selected those vehicle classes which have <b>more than 20 samples per class</b>.
          Following their experimental setup, we have <b>selected 3040 vehicle classes</b> and fine-tuned the Inception v3 CNN changing the number of nodes in last layer <b>from 1000 to 3040</b>.
          All the layers (earlier as well as latter) of the CNN have been trained. Since, there is a limitation in github for <b>uploading heavy files (maximum 100MB)</b>, we have uploaded the <b>trained weights to Google Drive https://bit.ly/2KR9Mne </b>
        </li>
        <li>
          The next step, was fine-tuning this CNN with Cars dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html). In this case, the number of nodes in last layer is <b>changed from 3040 to 196</b>.
          However, only the <b>latest fully connected layer is fine-tuned</b>. We have believed that the earlier layers of the network is already reach to low-level visual features from previous step. Therefore, there is no require to train again. 
          The trained <b>weights can be downloaded from Google Drive https://bit.ly/2WI889G </b>, due to previous github memory limit reason. 
        </li>
      </ul>
  </li>
  <li>
    <b>test.py</b> - contains the source code for testing the trained CNN model. It loads the trained model and reads all test samples with ImageDataGenerator. We have choosen as batch size 40, since, there are 8041 test samples  (8040 mod 40 = 0). This means that we have 201 (8041 / 40 = 201) feed forward iterations to the model. 
    For each test samples we have predicted top-5, top-3 and top-1 most probable vehicle classes (each vehicle class contains vehicle make, model and year). The <b>top-5, top-3 and top-1 accuracies</b> can be found in accuracy.txt file. 
  </li>
  <li>
    <b>inception_v3_cars_main.py</b> souce file is also used for testing, however, in comparison to test.py file, <b>it reads only one image file and produces top-5 most probable prediction results.</b> 
    <br/>
    The test sample with respective top-5 prediction result is illustrated in the following figure: 
    <p align="center"><img src="https://github.com/mkairanbay/grab/blob/master/top5.png" /></p>
  </li>
  <li>
    <b>accuracy.txt</b> file contains the top-5, top-3 and top-1 accuracies:<br/>
    <b>top-5 accuracy: 0.9587064676616915</b><br/>
    <b>top-3 accuracy: 0.9335820895522388</b><br/>
    <b>top-1 accuracy: 0.8105721393034826</b><br/>
  </li>
  <li>
    <b>output.txt</b> contains the output of test.py source code. Each line of the file characterized as follows (seperated by comma):
    <br/>
    <ul>
      <li>
        # of test cases (starting from 0)
      </li>
      <li>
        input image name. For example: 00076.jpg
      </li>
      <li>
        ground truth image label. For example: AM General Hummer SUV 200
      </li>
      <li>
        top five prediction results. For example: [AM General Hummer SUV 200,Geo Metro Convertible 199,Lamborghini Reventon Coupe 200,BMW 6 Series Convertible 200,Mazda Tribute SUV 201]
      </li>
      <li>
        top one prediction. For example: [AM General Hummer SUV 200]
      </li>
    </ul>
    The following line will show the first line of output.txt file<br/>
    <b>0,00076.jpg,AM General Hummer SUV 200,[AM General Hummer SUV 200,Geo Metro Convertible 199,Lamborghini Reventon Coupe 200,BMW 6 Series Convertible 200,Mazda Tribute SUV 201],[AM General Hummer SUV 200]</b><br>
   </li>
   <li>
      <b>cars.txt</b> the file contains enumerated class names
  </li>
</ul>

<br><br><br>
Using the result of this trained model, we have attempted to build web-service for smartphones which will predict the vehicle's make, model and year. 
The web-service can be found by following the url: http://lattaes.herokuapp.com/. <br>
<p align="center"><img src="https://github.com/mkairanbay/grab/blob/master/cars_main_page.PNG" /></p>
<br/>
User have to upload or take a photo of vehicle and press to "Predict" button. <br/>
<p align="center"><img src="https://github.com/mkairanbay/grab/blob/master/cars_second_page.PNG" /></p> <br>
It will upload the photo to the server and starts to predict the vehicle's make, model and year.  <br/>
<p align="center"><img src="https://github.com/mkairanbay/grab/blob/master/cars_loading_page.PNG" /> </p> <br/>
In the result it will show top-5 predicted classes with confidense scores. 
<br/>
<p align="center"><img src="https://github.com/mkairanbay/grab/blob/master/cars_result_page.PNG" /></p>
However, the performance of the web-service is not good. Because, the CNN is trained with Theano, however, the web-service is working with Tensorflow. The loading the Theano model to heroku takes very long time (more than 120 seconds), which restricts to the limitations of heroku. Therefore, we have decided to use tensorflow as a backend. However, it influenced to the performance of prediction in a bad way. In a feature, we have to solve this issue (train with tensorflow or convert the weights from theano to tensorflow) and upload proper weights.  
