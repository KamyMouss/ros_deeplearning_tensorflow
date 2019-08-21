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
