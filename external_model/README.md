## To launch

`roslaunch external_model start_image_recognition.launch`

## Notes

This package downloads a pre-built TensorFlow model for image recognition.

### image_recognition.py

```python
classify_image.maybe_download_and_extract()
```

This first line will download the **freezed-tensorflow-model**. A more exact model that was downloaded is **classify_image_graph_def.pb**, from the **imagenet_2012_challenge**. It's a model prepared for image recognition of hundreds of objects, trained with thousands of pre-labeled images.

in the file _imagenet_synset_to_human_label_map.txt_ you have a list of the objects that this model should be able to detect. On this list, you have the **encoding used by TensorFlow** and the **human-readable labels**.

As you can see, there are quite a few! This means that you could probably be able to use this for basic object recognition programs.

```python
self._session = tf.Session()
```

Here, you are starting a TensorFlow session. This will give you access to all of the TensorFlow functionality and you will be able to use the tensor soft-max from the downloaded model.

```python
classify_image.create_graph()
```

This will create a TensorFlow graph (more on this later).

```python
self._cv_bridge = CvBridge()
```

And here is our beloved CV_Bridge, which will allow us to use all of the power of OpenCV with ROS type images.

Let's also comment on the three lines that are important from a ROS point of view:

```python
self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
self._pub = rospy.Publisher('result', String, queue_size=1)
self.score_threshold = rospy.get_param('~score_threshold', 0.1)
```

Here, you declare a subscriber to an image topic, which we will remap to our particular robot camera's RGB topic.
You also declare a publisher, where you will write the highest-score object that was recognised.
And finally, you set the **score_threshold**. This you can increase up to 1.0. The higher the value, the most sure the detection has to be to consider it a correct and valid one. In this case, it's rather low to have loads of detections.

### classify_image.py

The only important variable for us is:

```python
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
```

This will set the TensorFlow compressed model and labeling that you choose to use. It will be downloaded and extracted in the **/tmp/imagenet** folder each time you run this. This is the way to go if you want other people to use your models and have the latest version of it. You can also download it to your package and use it from there. Less overhead each time, but not updated.

## Visualizing model with TensorBoard

One of the most important tools when doing any kind of Artificial Intelligence is visualising the model that you use and how the learning process is going.
with TensorFlow, we have **TensorBoard** to do so.

Sadly, TensorBoard needs to convert these .pb files to .log files to be able to read them easily. For that, you can use this handy script: _import_pb_to_tensorboard.py_

`python import_pb_to_tensorboard.py --model_dir=../course_tflow_image_student_data/show_case_pb_models/classify_image_graph_def.pb --log_dir=learning_logs`

Then run TensorBoard:

`tensorboard --logdir=learning_logs`

Whether you execute this locally or from a remote computer, you connect to TensorBoard through the browser. So, we need the address:

`public_ip`

To load TensorBoard in your browser, paste this:

`<public_ip_value>:6006`

And you will get a fantastic TensorBoard client.
Now, do you see that gray box that says "import?" Well, that's the model. To see what is inside, just double-click on it.

## Training your own Tensorflow Image Recognition Model

### Step 1: Labeling images

We will use a program that is called [LableImg](https://github.com/tzutalin/labelImg). This program generates .xml files based on images and how you label them. It makes the process less painful than writing the files by hand. To run:

`python3 /home/user/.labelImg/labelImg.py`

Now, just open any image with the GUI and you can start labeling by clicking the CreateRectagleBox, or by pressing the W key on the keyboard. Place these boxes where the object you want to label is, to mark it.

Once you have everything you want to train labeled, you have to save. This will generate an .xml file with the same name as your image.

Inside it, you have all of the information from the image, as well as the position, size, and label of each box. This will be used for the training of the TensorFlow model to validate the learning.

Each box is inside an object tag

One last thing you have to do is **COPY 10%** of the images into a **test** folder, and the other **90%** to a **train** folder. The percentages really depend on you. If you are unsure of whether the model works, put more images in the **test** folder.
It is **VERY IMPORTANT** that the images that are in the test folder **DON'T APPEAR** in the **train** folder. This guarantees that when testing, the training model is tested with images that it doesn't know. Otherwise, it won't learn correctly. It would be like testing students with the extact same exercise that you did yesterday in class... You won't know if they have learned or just have a good memory.

And that's it. Now, you just have to do the same thing for as many images as you need for the training. To make life a bit easier, you can import a full folder with all of the images inside.

### Step 2: Prepare the Image Data for the TensorFlow training

But, TensorFlow doesn't use .xml files. Instead, it uses a file type called .records. So, you have to convert your xml files.
This is divided into two main steps:

Convert XML to CSV
Convert CSV to RECORD

### Step 2.1: XML to CSV

The images are provided in the _images_ directory. You will use the _data_ directory afterwards for the CSV and RECORD files that are generated. To convert from **XML** to **CSV**, you have to run the **xml_to_csv.py** python script provided:

```python
# Convert XML to CSV files
roscd tf_unit1_pkg
python scripts/xml_to_csv.py
```

### Step 2.2: CSV to RECORD

Now to generate the Record files, you need to execute another python script:

In **extract_training_lables_csv.py**, you extract the labels that you are going to use. In this first example, the images only have **one label**, which is **mira_robot**. This will therefore, extract the label **mira_robot** from csv files generates previously.

And in **generate_tfrecord_n.py** file, there is a line that we have to know about:

```python
from object_detection.utils import dataset_util
```

Here you are importing from a folder called **object_detection**. To make this easier, you are going to download it into your package and set the python path to find it.

```bash
roscd tf_unit1_pkg
# Download the git models with the object detection module inside
rm -rf models
# We dont clone from git because its somewhat unstable , you can try though :git clone https://github.com/tensorflow/models.git
cp -r course_tflow_image_student_data/tf_models/models ./
# Compile protos messages python modules
cd models/research
protoc object_detection/protos/*.proto --python_out=.
echo "Check proto python files generated"
ls object_detection/protos/*_pb2.py
# We make all the modules inside models/research and also the slim folder inside available anywhere in python interpreter.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PYTHONPATH
roscd tf_unit1_pkg/scripts
```

Here you basically copy it from the repo we give you **course_tflow_image_student_data** and compile the **protobuffer** files to generate the python classes for each one. You don't need to know what they do, just know that you need them to be able to use this TensorFlow Library.

Ok, you can now execute the **python script generate_tfrecord.py**

```bash
roscd tf_unit1_pkg/scripts
python scripts/generate_tfrecord_n.py --image_path_input=images/train --csv_input=data/train_labels.csv  --output_path=data/train.record
python scripts/generate_tfrecord_n.py --image_path_input=images/test --csv_input=data/test_labels.csv  --output_path=data/test.record
```

This should generate the **.record** files for the **train** and **test** in the **data** folder.

### Step 2.3: Extra step: Check that the record files are ok

This is a step that is good to know how to do, although it's not necessary for the training. You have to know how to read what there is inside of a .record file, so that you can check other people's files and make sure that everything went ok.

For this, you have to launch a python script: **tfrecord_inspector.py**

It's simply converting the **record** file to a String. You can then save that output into a log file, like this:

```bash
# Check the Tf Record has been done
roscd tf_unit1_pkg
rm tfrecord_inspector.log
python scripts/tfrecord_inspector.py >> tfrecord_inspector.log
```

You can then read the **tfrecord_inspector.log** file. Beware that this can be a very big file because it also contains the pixel image data. We recommend that you use VIM to open it, it's the fastest way and it doesn't open all the data at once.
