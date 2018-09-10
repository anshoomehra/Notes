# This document is quick reference to key concepts and their definitions

- Terms:
  - **Numerical, Categorical**
  - **Discrete, Continuous**
  - **Nominal Numbers**: Nominal numbers or categorical numbers are numeric codes, meaning numerals used for labelling or identification only. The values of the numerals are irrelevant, and they do not indicate quantity, rank, or any other measurement. e.g. SSN
  - **Polynomial**:
  In mathematics, a polynomial is an expression consisting of variables (also called indeterminates) and coefficients, that involves only the operations of addition, subtraction, multiplication, and non-negative integer exponents of variables. An example of a polynomial of a single indeterminate x is x<sup>2</sup> − 4x + 7. An example in three variables is x<sup>3</sup> + 2xyz<sup>2</sup> − yz + 1
  - **Quadratic** : A quadratic equation is an equation of the second degree, meaning it contains at least one term that is squared. The standard form is ax² + bx + c = 0 with a, b, and c being constants, or numerical coefficients, and x is an unknown variable. One absolute rule is that the first constant “a” cannot be a zero.

**A loss function is a part of a cost function which is a type of an objective function.**

- **Loss Functions**: is usually a function defined on a data point, prediction and label, and measures the penalty.
  - **Step Function**
  - **L1, Absolute Loss**  (y - y^), used in linear regression
  - **L2, Square Loss** (y - y^)<sup>2</sup>, used in linear regression
  - **Hinge Loss** - used in SVM
  - **0/1 Loss** - used in theoretical analysis and definition of accuracy
  - **Sigmoid**
  - **Cross Entropy or Log Loss** - Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.
  - **Softmax**
  - **Logistic Regression**
<br><br>
- **Cost Function** is usually more general. It might be a sum of loss functions over your training set plus some model complexity penalty (regularization)
  - **Mean Absolute Error (MAE), L1**
  - **Mean Squared Error (MSE), L2**
  - etc..
<br><br>
- **Objective Function** is the most general term for any function that you optimize during training. For example, a probability of generating training set in maximum likelihood approach is a well defined objective function, but it is not a loss function nor cost function (however you could define an equivalent cost function). For example:
  - MLE is a type of objective function (which you maximize)
  - Divergence between classes can be an objective function but it is barely a cost function, unless you define something artificial, like 1-Divergence, and name it a cost
<br><br>
- **Why do Log ?**
Log-scale informs on relative changes (multiplicative), while linear-scale informs on absolute changes (additive). When do you use each? When you care about relative changes, use the log-scale; when you care about absolute changes, use linear-scale. This is true for distributions, but also for any quantity or changes in quantities.

  Note, I use the word "care" here very specifically and intentionally. Without a model or a goal, your question cannot be answered; the model or goal defines which scale is important. If you're trying to model something, and the mechanism acts via a relative change, log-scale is critical to capturing the behavior seen in your data. But if the underlying model's mechanism is additive, you'll want to use linear-scale.

  Example. Stock market.
  Stock A on day 1: $100. On day 2, $101. Every stock tracking service in the world reports this change in two ways! (1) +$1. (2) +1%. The first is a measure of absolute, additive change; the second a measure of relative change.

  Illustration of relative change vs absolute: Relative change is the same, absolute change is different
  Stock A goes from $1 to $1.10. Stock B goes from $100 to $110.

  Stock A gained 10%, stock B gained 10% (relative scale, equal)
  ...but stock A gained 10 cents, while stock B gained $10 (B gained more absolute dollar amount)

  If we convert to log space, relative changes appear as absolute changes.

  Stock A goes from log10($1) to log10($1.10) = 0 to .0413
  Stock B goes from log10($100) to log10($110) = 2 to 2.0413

  Now, taking the absolute difference in log space, we find that both changed by .0413.

  Both of these measures of change are important, and which one is important to you depends solely on your model of investing. There are two models. (1) Investing a fixed amount of principal, or (2) investing in a fixed number of shares.

  Model 1: Investing with a fixed amount of principal.

  Say yesterday stock A cost $1 per share, and stock B costs $100 a share. Today they both went up by one dollar to $2 and $101 respectively. Their absolute change is identical ($1), but their relative change is dramatically different (100% for A, 1% for B). Given that you have a fixed amount of principal to invest, say $100, you can only afford 1 share of B or 100 shares of A. If you invested yesterday you'd have $200 with A, or $101 with B. So here you "care" about the relative gains, specifically because you have a finite amount of principal.

  Model 2: fixed number of shares.

  In a different scenario, suppose your bank only lets you buy in blocks of 100 shares, and you've decided to invest in 100 shares of A or B. In the previous case, whether you buy A or B your gains will be the same ($100 - i.e. $1 for each share).

  Now suppose we think of a stock value as a random variable fluctuating over time, and we want to come up with a model that reflects generally how stocks behave. And let's say we want to use this model to maximize profit. We compute a probability distribution whose x-values are in units of 'share price', and y-values in probability of observing a given share price. We do this for stock A, and stock B. If you subscribe to the first scenario, where you have a fixed amount of principal you want to invest, then taking the log of these distributions will be informative. Why? What you care about is the shape of the distribution in relative space. Whether a stock goes from 1 to 10, or 10 to 100 doesn't matter to you, right? Both cases are a 10-fold relative gain. This appears naturally in a log-scale distribution in that unit gains correspond to fold gains directly. For two stocks whose mean value is different but whose relative change is identically distributed (they have the same distribution of daily percent changes), their log distributions will be identical in shape just shifted. Conversely, their linear distributions will not be identical in shape, with the higher valued distribution having a higher variance.

  If you were to look at these same distributions in linear, or absolute space, you would think that higher-valued share prices correspond to greater fluctuations. For your investing purposes though, where only relative gains matter, this is not necessarily true.
<br><br>
- **What is Phi/Chi Score, Z Score ?**
In statistics, the **phi coefficient** (or mean square contingency coefficient and denoted by φ or rφ) is a measure of association for two binary variables. Introduced by Karl Pearson,[1] this measure is similar to the Pearson correlation coefficient in its interpretation. In fact, a Pearson correlation coefficient estimated for two binary variables will return the phi coefficient.[2] The square of the phi coefficient is related to the chi-squared statistic for a 2×2 contingency table

  The **Chi Square statistic** is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.
  http://math.hws.edu/javamath/ryan/ChiSquare.html

  **z-score** is the number of standard deviations from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is. A z-score is also known as a standard score and it can be placed on a normal distribution curve. Z-scores range from -3 standard deviations (which would fall to the far left of the normal distribution curve) up to +3 standard deviations (which would fall to the far right of the normal distribution curve). In order to use a z-score, you need to know the mean μ and also the population standard deviation σ.
<br><br>
- **What is standard deviation, long tail, how do these impact learning ?**
The **standard deviation** provides some idea about the distribution of scores around the mean (average). The smaller the standard deviation, the more narrow the range between the lowest and highest scores or, more generally, that the scores cluster closely to the average score.

  You might think of it as a measure of "agreement" between raters. If everyone gave the same score, then the standard deviation would be zero and the agreement would be high (or perfect).

  From a more empirical perspective, we assume a normal distribution of scores (which is very true as the sample size increases). In this case it can be said that approximately two-thirds of the scores will fall within the range of plus or minus one standard deviation around the mean (and 90% of the scores would fall between two standard deviations, plus and minus). For example, with a mean score of 50 and standard deviation of 10, we would expect that most scores would fall between 40 and 60 and that nearly all scores would fall between 30 and 70.

  In a rough way, the standard deviation could be considered a measure of the extent to which one's observers agree or disagree with each another. The smaller the standard deviation suggests that people are in more agreement with one another than would be the case with a large standard deviation. Remember, however, that a single "outlying" response can distort the standard deviation and the sense of agreement between Observers. So taking this one important caveat into consideration, looking at the standard deviation can help leaders make a quick determination of whether others see them in the same fashion or not.

  The theory of the **Long Tail** is that our culture and economy is increasingly shifting away from a focus on a relatively small number of "hits" (mainstream products and markets) at the head of the demand curve and toward a huge number of niches in the tail.
<br><br>
- **Activation Functions**
  - **ReLU** - This activation function very simple, it says, if you are positive, I will return same value, if you are negative I will return 0. Another way to see ouput is between x & xero. This function is used a lot instead of sigmoid, as it can dramatically improve the performace of the network without sacrificing much accuracy. Since it's derivative is 1 for positive numbers. It is fascinating that this function which barely breaks linearilty can lead to such complex non-linear solutions.
  - **tanh** - Hyperbolic Tangent : This activation function though very similat to sigmoid has lager derivatives which provide great advances to NN.
  - **sigmoid**
  - **hard sigmoid** - Faster to compute than sigmoid
  - **ELU** - Exponential linear unit
  - **SELU** - Scaled Exponential Linear Unit
  - **Softplus**
  - **Softsign**
  - **linear**
  - **Advanced Activations**
    - **PReLU** - Parametric Rectified Linear Unit
    - **LeakyReLU**
    - **ThresholdedReLU**
<br><br>
 - **Overfitting, Regularization**
   - **Early Stopping**
   - **Ensemble**
   - **L1/L2**
   - **Injecting Noise**
   - **ReLU (Recified Linear Units)**  After each conv layer, it is convention to apply a nonlinear layer (or activation layer) immediately afterward.The purpose of this layer is to introduce nonlinearity to a system that basically has just been computing linear operations during the conv layers (just element wise multiplications and summations).In the past, nonlinear functions like tanh and sigmoid were used, but researchers found out that ReLU layers work far better because the network is able to train a lot faster (because of the computational efficiency) without making a significant difference to the accuracy. It also helps to alleviate the vanishing gradient problem, which is the issue where the lower layers of the network train very slowly because the gradient decreases exponentially through the layers (Explaining this might be out of the scope of this post, but see here and here for good descriptions). The ReLU layer applies the function f(x) = max(0, x) to all of the values in the input volume. In basic terms, this layer just changes all the negative activations to 0.This layer increases the nonlinear properties of the model and the overall network without affecting the receptive fields of the conv layer.

   - **Pooling** - The intuitive reasoning behind this layer is that once we know that a specific feature is in the original input volume (there will be a high activation value), its exact location is not as important as its relative location to the other features.

      As you can imagine, this layer <b>drastically reduces the spatial dimension</b> (the length and the width change but not the depth) of the input volume. This serves two main purposes:
      1. The first is that the amount of parameters or weights is reduced by x%, thus lessening the computation cost.
      2. The second is that it will control overfitting. This term refers to when a model is so tuned to the training examples that it is not able to generalize well for the validation and test sets. A symptom of overfitting is having a model that gets 100% or 99% on the training set, but only 50% on the test data.<br><br>

     - **POOLING CHOICES:**
       - **Max Pooling** [Max of filter is chosen]
       - **Avg Pooling**
       - **L2-Norm Pooling** <br> <br>

   - **Dropout** : dropout layers have a very specific function in neural networks. In the last section, we discussed the problem of overfitting, where after training, the weights of the network are so tuned to the training examples they are given that the network doesn’t perform well when given new examples. The idea of dropout is simplistic in nature. This layer “drops out” a random set of activations in that layer by setting them to zero. Simple as that. Now, what are the benefits of such a simple and seemingly unnecessary and counterintuitive process? Well, in a way, it forces the network to be redundant. By that I mean the network should be able to provide the right classification or output for a specific example even if some of the activations are dropped out. It makes sure that the network isn’t getting too “fitted” to the training data and thus helps alleviate the overfitting problem. An important note is that this layer is only used during training, and not during test time.


 - **Optimizers**
   - **ADAM**
   - **Adagrad**
   - etc
<br><br>
- **Feature Scaling** : https://stackoverflow.com/questions/26225344/why-feature-scaling
<br>
- **Pre-Processing**
  - Keras ImgaeDataGenerator : https://keras.io/preprocessing/image/
<br><br>
- **Data Augmentation**

- **Popular Networks** - Since 2010 ImageNet run yearly Large Scale Visual Competition achieveing better accuracy on object detection.

  - First Breaktrough was in 2012 - **AlexNeT** (Toronto) (Spotlight of Relu and Dropout)

  - 2014 - **VGG (Oxford) VGG16/VGG19** (Spotlight 3x3 Conv)

  - 2015 - Microsoft **ResNet** (Spotlight Massive Deep Arch, highly prone to vanishing gradient, avoided using skip-layer architetcure

- **Transfer Learning**

- **Autoencoders** (Encoder, Decoder Network : Compression, De-Noising  )

- **Weight Initialization Technique** :
(https://github.com/anshoomehra/udacity-deep-learning/blob/master/weight-initialization/weight_initialization.ipynb) The general rule for setting the weights in a neural network is to be close to zero without being too small. A good pracitce is to start your weights in the range of $[-y, y]$ where $y=1/\sqrt{n}$ ($n$ is the number of inputs to a given neuron).


  - **Uniform Uniform Distribution**: A uniform distribution has the equal probability of picking any number from a set of numbers
  - **Normal Distribution**: Unlike the uniform distribution, the normal distribution has a higher likelihood of picking number close to it's mean.
  - **Truncated Normal Distribution**: The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
<br><br>
- **Normalization**
  - **L1**
  - **L2** <br><br>

  **When to use L1 vs L2 ??** Both have different Sparsity rate as output.

  L1 Regularization produce more sparse vectors, so if we want to reduce number of weights & end up with small set we can use L1. It is also good for feature selection, sometimes we have hundreds of features, and with L1 technique we can select the ones whcih are important & turn rest in to zeros.

  L2 Regularization on the other hand, tends not to favor sparse veactors since it tries to maintain all the weights homogeneously small. This one normally gives better results for training models so it's the one we will use the most.

- **Process Stuck at Local Minima**, Mitigations
  - **Random Restart**
  - **Increase Momentum**
  <br><br>

- **Gradient Descent vs Stochastic Gradient Descent**
  <br><br>
  Gradient Descent we take one step with all the data and try to acheieve overall minima. The cost we pay is huge memory footprint and massive computational power.
<br><br>
  SGD in contrast, subset the data in n bathes and do n number of passes on each batch of data, which may not be as accurate as GD but it does take us in right direction progressively. The advantage here is that we can get sense of direction early on and also cost wrt memory and computational power neded is much less, offcourse we will apply SGD on overall data eventually (instead of one go & in batches).

- **Learning Rate Decay**: What learning rate to use, if we use high learning rate, we may miss local minima, however if we use small learning rate, we slow down the process. Best learning rate is the one whcih decrease as model gets closer to the solution.

- **Evaluation Metrics**
  - **Confusion Matrix** - Tells us how good our model is..
    - **Precision **- Ability of a classification model to return only relevant instances
    - **Recall** - Ability of a classification model to identify all relevant instances
    - **F1 (Harmonic Mean)**: single metric that combines recall and precision using the harmonic mean
    - **Receiver operating characteristic (ROC) curve**: plots the true positive rate (TPR) versus the false positive rate (FPR) as a function of the model’s threshold for classifying a positive
    - **Area under the curve (AUC)**: metric to calculate the overall performance of a classification model based on area under the ROC curve
<br><br>
- **Recurrent Neural Networks (RNNs)**
The neural network architectures you've seen so far were trained using the current inputs only. We did not consider previous(also known as temporal) inputs when generating the current output. In other words, our systems did not have any memory elements. RNNs address this very basic and important issue by using memory (i.e. past inputs to the network) when producing the current output.


  - **LSTM - Long Short-Term Memory (LSTM)** Cells were invented to tackle vanishing gradient, susequently fundamental issue of adding memory to these networks.LSTMs are very similar, with difference that it keep track of two memories Long Term and Short Term , so the present state depends on 3 inputs instead, i.e. Long Term Memory, Short Term Memory and Current Input all these 3 inputs gets merged to derive 3 outputs, prediction, and new Short Term + Long Term Memory. There are four main gates ..
    1. Learn Gate
    2. Forget Gate
    3. Remember Gate
    4. Use Gate
<br><br>
  - **GRU** - Gated Recurrent Unit It combines Forget & Learn gate into Update Gate & runs this through Combine Gate. It also working with only one Working Memory insteas of LTM and STM - but in practce this works very well.
