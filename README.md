<ul>
  <ul>
    This repository contains the source files for training and testing the Vehicle Make, Model and Year Prediction Task (https://www.aiforsea.com/computer-vision). 
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
          The next step, was fine-tuning this CNN with Cars dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html). In this case, the number of nodes in last layer is <b>chnaged from 3040 to 196</b>.
          However, only the <b>latest fully connected layer is only fine-tuned</b>. We have believed that the earlier layers of the network is already reach to low-level visual features from previous step which do not require to train again. 
          The rained <b>weights can be downloaded from Google Drive https://bit.ly/2WI889G </a>, due to previous github memory limit reason. 
        </li>
      </ul>
  </li>
  <li>
    <b>test.py</b> - contains the source code for testing the trained CNN model. It loads the trained model and reads all test samples with ImageDataGenerator (https://keras.io/preprocessing/image/).
    Since, there are 8041 test samples we have choosen as batch size 40 (8040 mod 40 = 0). This means that we have 201 feet forward iterations to the model. 
    For each test samples we have predicted top-5, top-3 and top-1 probable vehicle classes (vehicle make, model and year). The <a>top-5, top-3 and top-1 accuracies</b> can be found in accuracy.txt file. 
  </li>
  <li>
    <b>inception_v3_cars_main.py</b> souce file is also used for testing, however, in comparison to test.py file, <b>it read only one image file and produces top-5 most probable prediction results</b> 
  </li>
  <li>
    <b>accuracy.txt</b> file contains the top-5, top-3 and top-1 accuracies<br/>
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
        top five prediction. For example: [AM General Hummer SUV 200,Geo Metro Convertible 199,Lamborghini Reventon Coupe 200,BMW 6 Series Convertible 200,Mazda Tribute SUV 201]
      </li>
      <li>
        top one prediction. For example: [AM General Hummer SUV 200]
      </li>
    </ul>
    The following lines will show the first 3 lines of of output.txt file<br/>
    0,00076.jpg,AM General Hummer SUV 200,[AM General Hummer SUV 200,Geo Metro Convertible 199,Lamborghini Reventon Coupe 200,BMW 6 Series Convertible 200,Mazda Tribute SUV 201],[AM General Hummer SUV 200]<br>
    1,00457.jpg,AM General Hummer SUV 200,[AM General Hummer SUV 200,HUMMER H2 SUT Crew Cab 200,Jeep Wrangler SUV 201,HUMMER H3T Crew Cab 201,Jeep Liberty SUV 201],[AM General Hummer SUV 200]<br>
    2,00684.jpg,AM General Hummer SUV 200,[AM General Hummer SUV 200,Jeep Wrangler SUV 201,HUMMER H3T Crew Cab 201,Lamborghini Diablo Coupe 200,Volvo 240 Sedan 199],[AM General Hummer SUV 200]<br>
   </li>
    <ul>
      <li>
        <b>cars.txt</b> the file contains enumerated class names
      </li>
    </ul>
  </li>
</ul>
