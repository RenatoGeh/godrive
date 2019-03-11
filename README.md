GoDrive
=======

### Description

GoDrive is a lane following autonomous driving robot implementation
that uses image classification as a form of imitation learning.
[sum-product networks](http://spn.cs.washington.edu) (SPNs) are used as
a way to compute exact inference in linear time, allowing for accurate
and fast uncertainty measuring in real-time.

This code is part of my undergraduate thesis [Mobile Robot Self-Driving
Through Image Classification Using Discriminative Learning of
Sum-Product
Networks](https://www.ime.usp.br/~renatolg/mac0499/?lang=en).  The full
final thesis can be read
[here](https://www.ime.usp.br/~renatolg/mac0499/docs/thesis.pdf). Both
prediction and training of SPNs are done through the
[GoSPN](https://github.com/RenatoGeh/gospn) library.

### Objectives

The primary objectives of this implementation are twofold: both as a
comparative study on different SPN architectures and learning methods,
and also as a preliminar work on SPNs for self-driving and their
feasibility as a real-time prediction model. A third secondary objective
was to compare SPNs with state-of-the-art multilayer perceptrons (MLPs)
and convolutional neural-networks (CNNs).

### Lane following as self-driving

Ours is a primitive approach to self-driving, mainly that of lane
following through imitation learning. The robot's objective is to remain
inside a designated lane whilst still moving forward. Since the robot is
not allowed to stop, constantly moving forward, the prediction model
must be both accurate - identifying lane markings and making the
necessary heading corrections - and fast.

The robot was allowed to execute three different operations: go forward,
turn left, or turn right. Prediction output encoded these three commands
as a single byte.

### Hardware

We experimented with the Lego Mindstorm NXT, nicknamed Brick. A
Raspberry Pi Model 3, nicknamed Berry, was attached to the bot, together
with a low cost webcam. The Berry was then used for image capturing,
processing, and label prediction, sending the predicted label to the
Brick, who was only tasked with executing corresponding motor commands.

### Prediction

Prediction was done in real-time. The implementation contained in this
repository takes advantage of the Berry's four CPU cores. Three cores
are dedicated to computing each label concurrently. The fourth core is
used for image capturing, processing and sending the predicted byte to
the Brick.

### Training

Training was done separately on a desktop computer. You can read more
about training and validation
[here](https://www.ime.usp.br/~renatolg/mac0499/docs/thesis.pdf)

### Results

Results are available both in the
[thesis](https://www.ime.usp.br/~renatolg/mac0499/docs/thesis.pdf) and
also in [video](https://www.ime.usp.br/~renatolg/mac0499/video.html?lang=en).

### Code structure

The code is structured as follows:

- `root`:
    * `contest.go`: connection test for testing if the bot is visible
    * `main.go`: main file for executing training or self-driving
- `bot`:
    * `bot.go`: bot loop, communication, prediction and image processing
    * `usb.go`: USB communication handling from Berry to Brick
- `camera`:
    * `writer.go`: writer inferface for recording camera feed
    * `camera.go`: camera processing and capturing
    * `trans.go`: image transformations
- `data`:
    * `data.go`: pulling data from dataset
- `models`:
    * `model.go`: interface for prediction models
    * `accuracy.go`: validation code for accuracy measuring
    * `dennis.go`: Dennis-Ventura architecture inference, learning and serialization
    * `gens.go`: Gens-Domingos architecture inference, learning and serialization
- `java`:
    * `Remote.java`: Brick-side code for interpreting messages as motor power
