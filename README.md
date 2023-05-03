Download Link: https://assignmentchef.com/product/solved-cs189-introduction-to-machine-learning-hw6
<br>
<ol>

 <li>Submit your predictions for the test sets to Kaggle as early as possible. Include your Kaggle scores in your write-up (see below). The Kaggle competition for this assignment can be found at

  <ul>

   <li><a href="https://www.kaggle.com/t/b500e3c2fb904ed9a5699234d3469894">https://www.kaggle.com/t/b500e3c2fb904ed9a5699234d3469894</a></li>

  </ul></li>

 <li>Submit a PDF of your homework, with an appendix listing all your code, to the Gradescope assignment entitled “Homework 6 Write-Up”. In addition, please include, as your solutions to each coding problem, the specific subset of code relevant to that part of the problem. You may typeset your homework in LaTeX or Word (submit PDF format, not .doc/.docx format) or submit neatly handwritten and scanned solutions. Please start each question on a new page. If there are graphs, include those graphs in the correct sections. Do not put them in an appendix. We need each solution to be self-contained on pages of its own.

  <ul>

   <li>In your write-up, please state with whom you worked on the homework.</li>

   <li>In your write-up, please copy the following statement and sign your signature next to it. (Mac Preview and FoxIt PDF Reader, among others, have tools to let you sign a PDF file.) We want to make it <em>extra </em>clear so that no one inadvertently cheats.</li>

  </ul></li>

</ol>

<em>“I certify that all solutions are entirely in my own words and that I have not looked at another student’s solutions. I have given credit to all external sources I consulted.”</em>

<ol start="3">

 <li>Submit all the code needed to reproduce your results to the Gradescope assignment entitled “Homework 6 Code”. Yes, you must submit your code twice: in your PDF write-up following the directions as described above so the readers can easily read it, and once in compilable/interpretable form so the readers can easily run it. Do NOT include any data files we provided. Please include a short file named README listing your name, student ID, and instructions on how to reproduce your results. Please take care that your code doesn’t take up inordinate amounts of time or memory. If your code cannot be executed, your solution cannot be verified.</li>

</ol>

In this assignment, you will develop neural network models with MDS189. Many toy datasets in machine learning (and computer vision) serve as excellent tools to help you develop intuitions about methods, but they cannot be directly used in real-world problems. MDS189 could be.

Under the guidance of a strength coach here at UC Berkeley, we modeled the movements in MDS189 after the real-world <a href="https://www.functionalmovement.com/">Functional Movement Screen</a> (FMS). The FMS has 7 different daily movements, and each is scored according to a specific 0-3 rubric. Many fitness and health-care professionals, such as personal trainers and physical therapists, use the FMS as a diagnostic assessment of their clients and athletes. For example, there is a large body of research that suggests that athletes whose cumulative FMS score falls below 14 have a higher risk of injury. In general, the FMS can be used to assess functional limitations and asymmetries. More recent research has begun investigating the relationship between FMS scores and fall risk in the elderly population.

In modeling MDS189 after the real-world Functional Movement Screen, we hope the insight you gain from the experience of collecting data, training models, evaluating performance, etc. will be meaningful.

A large part of this assignment makes use of MDS189. Thank you to those who agreed to let us use your data in MDS189! Collectively, you have enabled everyone to enjoy the hard-earned reward of data collection.

Download MDS189 immediately. At 3GB+ of data, MDS189 is rather large, and it will require a while to download. You can access MDS189 through <a href="https://forms.gle/rTkYZuCD9jhDfcX99">this Google form</a><a href="https://forms.gle/rTkYZuCD9jhDfcX99">.</a> When you gain access to MDS189, you are required to agree that you will not share MDS189 with anyone else. <em>Everyone </em>must fill out this form, and sign the agreement. If you use MDS189 without signing the agreement, you (and whomever shared the data with you) will receive an automatic zero on all the problems on this homework relating to MDS189.

The dataset structure for MDS189 is described in mds189format.txt, which you will be able to find in the Google drive folder.

<h1>1          Data Visualization</h1>

When you begin to work with a new dataset, one of the first things you should do is spend some time visualizing the data. For images, you must look at the pixels to help guide your intuitions while developing models. Pietro Perona, a computer vision professor at Caltech, has said that when you begin working with a new dataset, “you should spend two days just looking at the data.” We do not recommend you spend quite that much time looking at MDS189; the point is that the value of quality time spent visualizing a new dataset cannot be overstated.

We provide several visualization tools in mds189visualize.ipynb that will enable you to view montages of: key frames, other videos frames, ground truth keypoints (i.e., what you labeled in LabelBox), automatically detected keypoints from <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">OpenPose</a><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">,</a> and bounding boxes based on keypoint detections.

Note: Your responses to the questions in this problem should be at most two sentences.

<ul>

 <li>To get a sense of the per-subject labeling quality, follow the Part 1: Same subject instructions in the cell titled Key Frame visualizations. For your write-up, you do not need to include any images from your visualizations. You do need to include answers to the following questions (these can be general statements, you are not required to reference specific subject ids):

  <ol>

   <li>What do you observe about the quality of key frame annotations? Pay attention to whether the key frames reflect the movement labeled.</li>

   <li>What do you observe about the quality of keypoint annotations? Pay attention to things like: keypoint location and keypoint colors, which should give a quick indication of whether a labeled keypoint corresponds to the correct body joint.</li>

  </ol></li>

 <li>To quickly get a sense of the overall variety of data, follow the Part 2: Random subject instructions in the cell titled Key Frame visualizations. Again, for your write-up, you do not need to include any images from your visualizations. Include an answer to the following question:

  <ol>

   <li>What do you observe about the variety of data? Pay attention to things like differences in key frame pose, appearance, lighting, frame aspect ratio, etc.</li>

  </ol></li>

 <li>We ran the per-frame keypoint detector OpenPose on your videos to estimate the pose in your video frames. Based on these keypoints, we also estimated the bounding box coordinates for a rectangle enclosing the detected subject. Follow the Part 3: same subject instructions in the cell titled Video Frame visualizations. Again, for your write-up, you do not need to include any images from your visualizations. You do need to include answers to the following question:

  <ol>

   <li>What do you observe about the quality of bounding box and OpenPose keypoint annotations? Pay attention to things like annotation location, keypoint colors, number of people detected, etc.</li>

   <li>Based on the third visualization, where you are asked to look at all video frames for on movement, what do you observe about the sampling rate of the video frames? Does it appear to reasonably capture the movement?</li>

  </ol></li>

 <li>For the key frames, we can take advantage of the knowledge that the poses should be similar to the labeled poses in heatherlckwd’s key frames. Using <a href="https://en.wikipedia.org/wiki/Procrustes_analysis">Procrustes analysis</a><a href="https://en.wikipedia.org/wiki/Procrustes_analysis">,</a> we aligned each key frame pose with the corresponding key frame pose from heatherlckwd. Compare the plot of the raw Neck keypoints with the plot of the (normalized) aligned Neck keypoints. What do you observe?</li>

</ul>

Note: We introduce the aligned poses because we offer them as a debugging tool to help you develop neural network code in problem 2. Your reported results cannot use the aligned poses as training data.

<h1>2          Modular Fully-Connected Neural Networks</h1>

First, we will establish some notation for this problem. We define

<em>h<sub>i</sub></em><sub>+1 </sub>= σ(<em>z<sub>i</sub></em>) = σ(<em>W<sub>i</sub>h<sub>i </sub></em>+ <em>b<sub>i</sub></em>).

In this equation, <em>W<sub>i </sub></em>is an <em>n<sub>i</sub></em><sub>+1</sub>×<em>n<sub>i </sub></em>matrix that maps the input <em>h<sub>i </sub></em>of dimension <em>n<sub>i </sub></em>to a vector of dimension <em>n<sub>i</sub></em><sub>+1</sub>, where <em>n<sub>i</sub></em><sub>+1 </sub>is the size of layer <em>i </em>+ 1. The vector <em>b<sub>i </sub></em>is the bias vector added after the matrix multiplication, and σ is the nonlinear function applied element-wise to the result of the matrix multiplication and addition. <em>z<sub>i </sub></em>= <em>W<sub>i</sub>h<sub>i </sub></em>+<em>b<sub>i </sub></em>is a shorthand for the intermediate result within layer <em>i </em>before applying the activation function σ. Each layer is computed sequentially where the output of one layer is used as the input to the next. To compute the derivatives with respect to the weights <em>W<sub>i </sub></em>and the biases <em>b<sub>i </sub></em>of each layer, we use the chain rule starting with the output of the network and propagate backwards through the layers, which is where the backprop algorithm gets its name.

In this problem, we will implement fully-connected networks with a modular approach. This means different layer types are implemented individually, which can then be combined into models with different architectures. This enables code re-use, quick implementation of new networks and easy modification of existing networks.

<h2>2.1          Layer Implementations</h2>

Each layer’s implementation will have two defining functions:

<ol>

 <li>forward This function has as input the output <em>h<sub>i </sub></em>from the previous layer, and any relevant parameters, such as the weights <em>W<sub>i </sub></em>and bias <em>b<sub>i</sub></em>. It returns an output <em>h<sub>i</sub></em><sub>+1 </sub>and a cache object that stores intermediate values needed to compute gradients in the backward pass.</li>

</ol>

<table width="596">

 <tbody>

  <tr>

   <td width="596">def forward(h, w):“”” example forward function skeleton code with h: inputs, w: weights””” # Do computations…z = # Some intermediate output # Do more computations…out = # the output cache = (h, w, z, out) # Values needed for gradient computation return out, cache</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>backward This function has input: upstream derivatives and the cache object. It returns the local gradients with respect to the inputs and weights.</li>

</ol>

<table width="596">

 <tbody>

  <tr>

   <td width="596">def backward(dout, cache):“”” example backward function skeleton code with dout: derivative of loss with respect to outputs and,→ cache from the forward pass “””# Unpack cache h, w, z, out = cache# Use values in cache, along with dout to compute derivatives dh = # Derivative of loss with respect to a dw = # Derivative of loss with respect to w return dh, dw</td>

  </tr>

 </tbody>

</table>

Your layer implementations should go into the provided layers.py script. The code is clearly marked with TODO statements indicating what to implement and where.

When implementing a new layer, it is important to manually verify correctness of the forward and backward passes. Typically, the gradients in the backward pass are checked against numerical gradients. We provide a test script startercode.ipynb for you to use to check each of layer implementations, which handles the gradient checking. Please see the comments of the code for how to appropriately use this script.

In your write-up, provide the following for each layer you’ve implemented.

<ol>

 <li>Listings of (the relevant parts of) your code.</li>

 <li>Written justification/derivation for the derivatives in your backward pass for <em>all </em>the layers that you implement.</li>

 <li>The output of running numerical gradient checking.</li>

 <li>Answers to any inline questions.</li>

</ol>

2.1.1       Fully-Connected (fc) Layer

In layers.py, you are to implement the forward and backward functions for the fully-connected layer. The fully-connected layer performs an a<a href="https://en.wikipedia.org/wiki/Affine_transformation">ffi</a><a href="https://en.wikipedia.org/wiki/Affine_transformation">ne transformation</a> of the input: fc(<em>h</em>) = <em>Wa </em>+ <em>b</em>. Write your fc layer for a general input <em>h </em>that contains a mini-batch of <em>B </em>examples, each of which is of shape (<em>d</em><sub>1</sub>,··· ,<em>d<sub>k</sub></em>).

2.1.2      Activation Functions

In layers.py, implement the forward and backward passes for the ReLU activation function



0     γ &lt; 0

σReLU(γ) = γ otherwise

Note that the activation function is applied element-wise to a vector input.

There are many other activation functions besides ReLU, and each activation function has its advantages and disadvantages. One issue commonly seen with activation functions is vanishing gradients, i.e., getting zero (or close to zero) gradient flow during backpropagation. Which of activation functions (among: linear, ReLU, tanh, sigmoid) experience this problem? Why? What types of one-dimensional inputs would lead to this behavior?

2.1.3      Softmax Loss

In subsequent parts of this problem, we will train a network to classify the movements in MDS189. Therefore, we will need the softmax loss, which is comprised of the softmax activation followed by the crossentropy loss. It is a minor technicality, but worth noting that the softmax is just the squashing function that enables us to apply the cross-entropy loss. Nevertheless, it is a commonly used shorthand to refer to this as the softmax loss.

The softmax function has the desirable property that it outputs a probability distribution. For this reason, many classification neural networks use the softmax. Technically, the softmax activation takes in <em>C </em>input numbers and outputs <em>C </em>scores which represents the probabilities for the sample being in each of the possible <em>C </em>classes. Formally, suppose <em>s</em><sub>1 </sub>··· <em>s<sub>C </sub></em>are the <em>C </em>input scores; the outputs of the softmax activations are

<em>e</em><em>s</em><em>i</em>

<em>t</em><em>i </em>= P<em>Ck</em>=1 <em>e</em><em><sub>s</sub></em><em><sub>k</sub></em>

for <em>i </em>∈ [1,<em>C</em>]. The cross-entropy loss is

<em>E </em>= −log<em>t<sub>c</sub></em>,

where <em>c </em>is the correct label for the current example.

Since the loss is the last layer within a neural network, and the backward pass of the layer is immediately calculated after the foward pass, layers.py merges the two steps with a single function called softmaxloss.

You have to be careful when you implement this loss, otherwise you will run into issues with numerical stability. Let <em>m </em>= max<em><sup>C</sup><sub>i</sub></em><sub>=1 </sub><em>s<sub>i </sub></em>be the max of the <em>s<sub>i</sub></em>. Then

<em>E </em>= −log<em>t</em><em>c </em>= log P<em>Ce</em><em>s</em><em>c </em><em>s</em><em>k </em>= log P<em>Ce</em><em>s</em><em>c</em>−<em>e</em><em>ms</em><em>k</em>−<em>m </em>= −(<em>s</em><em>c </em>− <em>m</em>) + logX<em>kC</em>=1 <em>e</em><em>s</em><em>k</em>−<em>m</em>. <em>e</em>

<em>k</em>=1                         <em>k</em>=1

We recommend using the rightmost expression to avoid numerical problems.

Finish the softmax loss in layers.py.

<h2>2.2          Two-layer Network</h2>

Now, you will use the layers you have written to implement a two-layer network (also referred to as a one <em>hidden </em>layer network) that classifies movement type based on keypoint annotations. The input features are pre-processed keypoint annotations of an image, and the output are one of 8 possible movement types: deadbug, hamstrings, inline, lunge, stretch, pushup, reach, or squat. You should implement the following network architecture: input – fc layer – ReLU activation – fc layer – softmax loss. Implement the class FullyConnectedNet in fcnet.py. Note that this class supports multi-layer networks, not just two-layer networks. You will need this functionality in the next part. In order to train your model, you need two other components, listed below.

<ol>

 <li>The data loader, which is responsible for loading batches of data that will be fed to your model during training. Data pre-processing should be handled by the data loader.</li>

 <li>The solver, which encapsulates all the logic necessary for training models.</li>

</ol>

You don’t need to worry about those, since they are already implemented for you. See startercode.ipynb for an example.

For your part, you will need to instantiate a model of your two-layer network, load your training and validation data, and use a Solver instance to train your model. Explore different hyperparameters including the learning rate, learning rate decay, batch size, the hidden layer size, and the weight scale initialization for the parameters. Report the results of your exploration, including what parameters you explored and which set of parameters gave the best validation accuracy.

Debugging note: The default data loader returns raw poses, i.e., the ones that you labeled in LabelBox. As a debugging tool only, you can replace this with the heatherlckwd-aligned, normalized poses. It’s easier and faster to get better performance with the aligned poses. Use this for debugging only! You can use this feature by setting debug = True in the starter code. All of your reported results must use the un-aligned, raw poses for training data.

<h2>2.3          Multi-layer Network</h2>

Now you will implement a fully-connected network with an arbitrary number of hidden layers. Use the same code as before and try different number of layers (1 hidden layer to 4 hidden layers) as well as different number of hidden units. Include in your write-up what kinds of models you have tried, their hyperparameters, and their training and validation accuracies. Report which architecture works best.

<h1>3          Convolution and Backprop Revisited</h1>

In this problem, we will explore how image masking can help us create useful high-level features that we can use instead of raw pixel values. We will walk through how discrete 2D convolution works and how we can use the backprop algorithm to compute derivatives through this operation.

<ul>

 <li>To start, let’s consider convolution in one dimension. Convolution can be viewed as a function that takes a signal <em>I</em>[] and a mask <em>G</em>[], and the discrete convolution at point <em>t </em>of the signal with the mask is</li>

</ul>

∞

X

(<em>I </em>∗ <em>G</em>)[<em>t</em>] =             <em>I</em>[<em>k</em>]<em>G</em>[<em>t </em>− <em>k</em>]

<em>k</em>=−∞

If the mask <em>G</em>[] is nonzero in only a finite range, then the summation can be reduced to just the range in which the mask is nonzero, which makes computing a convolution on a computer possible.

Figure 1: Figure showing an example of one convolution.

As an example, we can use convolution to compute a derivative approximation with finite differences. The derivative approximation of the signal is <em>I</em><sup>0</sup>[<em>t</em>] ≈ (<em>I</em>[<em>t </em>+ 1] − <em>I</em>[<em>t </em>− 1])/2. Design a mask <em>G</em>[] such that (<em>I </em>∗ <em>G</em>)[<em>t</em>] = <em>I</em><sup>0</sup>[<em>t</em>].

<ul>

 <li>Convolution in two dimensions is similar to the one-dimensional case except that we have an additional dimension to sum over. If we have some image <em>I</em>[<em>x</em>,<em>y</em>] and some mask <em>G</em>[<em>x</em>,<em>y</em>], then the convolution at the point (<em>x</em>,<em>y</em>) is</li>

</ul>

∞               ∞ X X

(<em>I </em>∗ <em>G</em>)[<em>x</em>,<em>y</em>] =                          <em>I</em>[<em>m</em>,<em>n</em>]<em>G</em>[<em>x </em>− <em>m</em>,<em>y </em>− <em>n</em>]

<em>m</em>=−∞ <em>n</em>=−∞

or equivalently,

∞               ∞ X X

(<em>I </em>∗ <em>G</em>)[<em>x</em>,<em>y</em>] =                          <em>G</em>[<em>m</em>,<em>n</em>]<em>I</em>[<em>x </em>− <em>m</em>,<em>y </em>− <em>n</em>],

<em>m</em>=−∞ <em>n</em>=−∞

because convolution is commutative.

In an implementation, we’ll have an image <em>I </em>that has three color channels <em>I<sub>r</sub></em>, <em>I<sub>g</sub></em>, <em>I<sub>b </sub></em>each of size <em>W </em>× <em>H </em>where <em>W </em>is the image width and <em>H </em>is the height. Each color channel represents the intensity of red, green, and blue for each pixel in the image. We also have a mask <em>G </em>with finite support. The mask also has three color channels, <em>G<sub>r</sub></em>,<em>G<sub>g</sub></em>,<em>G<sub>b</sub></em>, and we represent these as a <em>w </em>× <em>h </em>matrix where <em>w </em>and <em>h </em>are the width and height of the mask. (Note that usually <em>w </em> <em>W </em>and <em>h </em> <em>H</em>.) The output (<em>I </em>∗ <em>G</em>)[<em>x</em>,<em>y</em>] at point

(<em>x</em>,<em>y</em>) is

<em>w</em>−1 <em>h</em>−1

XX X

(<em>I </em>∗ <em>G</em>)[<em>x</em>,<em>y</em>] =                                 <em>I<sub>c</sub></em>[<em>x </em>+ <em>a</em>,<em>y </em>+ <em>b</em>] · <em>G<sub>c</sub></em>[<em>a</em>,<em>b</em>]

<em>a</em>=0 <em>b</em>=0 <em>c</em>∈{<em>r</em>,<em>g</em>,<em>b</em>}

In this case, the size of the output will be (1 + <em>W </em>− <em>w</em>) × (1 + <em>H </em>− <em>h</em>), and we evaluate the convolution only within the image <em>I</em>. (For this problem we will not concern ourselves with how to compute the convolution along the boundary of the image.) To reduce the dimension of the output, we can do a strided convolution in which we shift the convolutional mask by <em>s </em>positions instead of a single position, along the image. The resulting output will have size b1 + (<em>W </em>− <em>w</em>)/<em>s</em>c × b1 + (<em>H </em>− <em>h</em>)/<em>s</em>c.

Write pseudocode to compute the convolution of an image <em>I </em>with a set of masks <em>G </em>and a stride of <em>s</em>. Hint: to save yourself from writing low-level loops, you may use the operator ∗ for element-wise

Figure 2: Figure showing an example of one maxpooling.

multiplication of two matrices (which is not the same as matrix multiplication) and invent other notation when convenient for simple operations like summing all the elements in the matrix.

<ul>

 <li>Masks can be used to identify different types of features in an image such as edges or corners. Design a mask <em>G </em>that outputs a large value for vertically oriented edges in image <em>I</em>. By “edge,” we mean a vertical line where a black rectangle borders a white rectangle. (We are not talking about a black line with white on both sides.)</li>

 <li>Although handcrafted masks can produce edge detectors and other useful features, we can also learn masks (sometimes better ones) as part of the backpropagation algorithm. These masks are often highly specific to the problem that we are solving. Learning these masks is a lot like learning weights in standard backpropagation, but because the same mask (with the same weights) is used in many different places, the chain rule is applied a little differently and we need to adjust the backpropagation algorithm accordingly. In short, during backpropagation each weight <em>w </em>in the mask has a partial derivative <sub>∂</sub><u><sup>∂</sup></u><em><sub>w</sub><u><sup>L </sup></u></em>that receives contributions from every patch of image where <em>w </em>is applied.</li>

</ul>

Let <em>L </em>be the loss function or cost function our neural network is trying to minimize. Given the input image <em>I</em>, the convolution mask <em>G</em>, the convolution output <em>R </em>= <em>I</em>∗<em>G</em>, and the partial derivative of the error with respect to each scalar in the output, <sub>∂<em>R</em></sub><u><sup>∂</sup></u><sub>[</sub><em><u><sup>L</sup></u><sub>i</sub></em>,<em><sub>j</sub></em><sub>]</sub>, write an expression for the partial derivative of the loss with respect to a mask weight, <sub>∂<em>G</em></sub><sup>∂</sup><em>c</em><sub>[</sub><em><sup>L</sup><sub>x</sub></em>,<em><sub>y</sub></em><sub>]</sub>, where <em>c </em>∈ {<em>r</em>,<em>g</em>,<em>b</em>}. Also write an expression for the derivative of

<u>∂<em>L</em></u>

∂<em>I<sub>c</sub></em>[<em>x</em>,<em>y</em>].

<ul>

 <li>Sometimes, the output of a convolution can be large, and we might want to reduce the dimensions of the result. A common method to reduce the dimension of an image is called max pooling. This method works similar to convolution in that we have a mask that moves around the image, but instead of multiplying the mask with a subsection of the image, we take the maximum value in the subimage. Max pooling can also be thought of as downsampling the image but keeping the largest activations for each channel from the original input. To reduce the dimension of the output, we can do a strided max pooling in which we shift the max pooling mask by <em>s </em>positions instead of a single position, along the input. Given a mask size of <em>w </em>× <em>h</em>, and a stride <em>s</em>, the output will be b1 + (<em>W </em>− <em>w</em>)/<em>s</em>c × b1 + (<em>H </em>− <em>h</em>)/<em>s</em>c for an input image of size <em>W </em>× <em>H</em>.</li>

</ul>

Let the output of a max pooling operation be an array <em>R</em>. Write a simple expression for element <em>R</em>[<em>i</em>, <em>j</em>] of the output.

<ul>

 <li>Explain how we can use the backprop algorithm to compute derivates through the max pooling operation. (A plain English answer will suffice; equations are optional.)</li>

</ul>

<h1>4          Convolutional Neural Networks (CNNs)</h1>

In this problem we will revisit the problem of classifying movements based on the key frames. The fullyconnected networks we have worked with in the previous problem have served as a good testbed for experimentation because they are very computationally efficient. However, in practice state-of-the-art methods on image data use convolutional networks.

It is beyond the scope of this class to implement an efficient forward and backward pass for convolutional layers. Therefore, it is at this point that we will leave behind your beautiful code base from problem 1 in favor of developing code for this problem in the popular deep learning framework PyTorch.

PyTorch executes dynamic computational graphs over Tensor objects that behave similarly to numpy ndarray. It comes with a powerful automatic differentiation engine that removes the need for manual backpropagation. You should install PyTorch and take a look at the basic tutorial here: <a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">https://pytorch.org/ </a><a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">tutorials/beginner/deep_learning_60min_blitz.html</a><a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">.</a> The installation instructions can be found at <a href="https://pytorch.org/">https://pytorch.org/</a> under ‘Quick Start Locally’. You will be able to specify your operating system and package manager (e.g., pip or conda).

Debugging notes

<ol>

 <li>One of the most important debugging tools when training a new network architecture is to train the network first on a small set of data, and verify that you can overfit to that data. This could be as small as a single image, and should not be more than a batch size of data.</li>

 <li>You should see your training loss decrease steadily. If your training loss starts to increase rapidly (or even steadily), you likely need to decrease your learning rate. If your training loss hasn’t started noticeably decreasing within one epoch, your model isn’t learning anything. In which case, it may be time to either: a) change your model, or b) increase your learning rate.</li>

 <li>It can be helpful to save a log file for each model that contains the training loss for each <em>N </em>steps, and the validation loss for each <em>M </em>&gt;&gt; <em>N </em> This way, you can plot the loss curve vs number of iterations, and compare the loss curves between models. It can help speed up the comparison between model performances.</li>

 <li>Do not delete a model architecture you have tried from the code. Often, you want the flexibility to run any model that you have experimented with at any time without a re-coding effort.</li>

 <li>Keep track of the model architectures you run, save each model’s weights, and record the evaluation scores for each model. For example, you could record this information in a spreadsheet with structure: model architecture info (could be as simple as the name of the model used in the code), accuracy for each of the 8 classes, average accuracy across all 8 classes, and location of the model weights.</li>

</ol>

These networks take time to train. Please start early!

Cloud credits. Training on a CPU is much slower than training on a GPU. We don’t want you to be limited by this. You have a few options for training on a GPU:

<ol>

 <li>Google has generously provided $50 in cloud credits for each student in our class. This is exclusively for students in CS 189/289A. Please do not share <a href="https://google.secure.force.com/GCPEDU?cid=oHBFmmD32%2B27X4XRqTw8qFekj9f4dKMBp757cBmUbt7L%2B0vyYrVi07lRfWvBkkVO">this</a> link outside of this class. We were only given enough cloud credits for each student in the class to get one $50 credit. Please be reasonable.</li>

 <li>Google Cloud gives first-time users $300 in free credits, which anyone can access at <a href="https://cloud.google.com/">https:// </a><a href="https://cloud.google.com/">google.com/</a></li>

 <li>(least user-friendly) Amazon Web Services gives first-time users $100 in free credits, which anyone can access at <a href="https://aws.amazon.com/education/awseducate/">https://aws.amazon.com/education/awseducate/</a></li>

 <li>(most user-friendly) Google Colab, which interfaces with Google drive, operates similarly to Jupyter notebook, and offers free GPU use for anyone at <a href="https://colab.research.google.com/">https://colab.research.google.com/</a> Google Colab also offers some nice <a href="https://medium.com/looka-engineering/how-to-use-tensorboard-with-pytorch-in-google-colab-1f76a938bc34">tools</a> for visualizing training progress (see debugging note 3 above).</li>

</ol>

<ul>

 <li>Implement a CNN that classifies movements based on a single key frame as input. We provide skeleton code in problem4, which contains the fully implemented data loader (mds189.py) and the solver (in train.py). For your part, you are to write the model, the loss, and modify the evaluation. There are many TODO and NOTE statements in problem4/train.py to help guide you. Experiment with a few different model architectures, and report your findings.</li>

 <li>For your best CNN model, plot the training and validation loss curves as a function of number of steps.</li>

 <li>Draw the architecture for your best CNN model. How do the number of parameters compare between your best CNN and a comparable architecture in which you replace all convolutional layers with fullyconnected layers?</li>

 <li>Train a movement classification CNN with your best model architecture from part (a) that now takes as input a random video frame, instead of a key frame. Note: there are many more random frames than there are key frames, so you are unlikely to need as many epochs as before.</li>

 <li>Compare your (best) key frame and (comparable architecture) random frame CNN performances by showing their per-movement accuracy in a two-row table. Include their overall accuracies in the table.</li>

 <li>When evaluating models, it is important to understand your misclassifications and error modes. For your random image and key frame CNNs, plot the confusion matrices. What do you observe? For either CNN, visualize your model’s errors, i.e., look at the images and/or videos where the network misclassifies the input. What do you observe about your model’s errors? Be sure to clearly state which model you chose to explore.</li>

 <li>For the <a href="https://www.kaggle.com/t/b500e3c2fb904ed9a5699234d3469894">Kaggle</a> competition, you will evaluate your best CNN trained for the task of movement classification based on a random video frame as input. In part (d), we did not ask you to tune your CNN in any way for the video frame classifier. For your Kaggle submission, you are welcome to make any improvements to your CNN. The test set of images is located in the testkaggleframes directory in the dataset Google drive folder. For you to see the format of the Kaggle submission, we provide the sample file kagglesubmissionformat.csv, where the predictedlabels should be replaced with your model’s prediction for the movement, e.g., reach, squat, inline, lunge, hamstrings, stretch, deadbug, or pushup.</li>

</ul>