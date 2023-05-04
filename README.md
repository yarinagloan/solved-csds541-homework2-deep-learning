Download Link: https://assignmentchef.com/product/solved-cs_ds541-homework2-deep-learning
<br>
<ol>

 <li><strong>XOR problem </strong>[, on paper]: Show (by deriving the gradient, setting to 0, and solving mathematically, not in Python) that the values for <strong>w </strong>= (<em>w</em><sub>1</sub><em>,w</em><sub>2</sub>) and <em>b </em>that minimize the function <em>J</em>(<strong>w</strong><em>,b</em>) in Equation 6.1 (in the <em>Deep Learning </em>textbook) are: <em>w</em><sub>1 </sub>= 0, <em>w</em><sub>2 </sub>= 0, and <em>b </em>= 0<em>.</em></li>

 <li><em>L</em><sub>2</sub><strong>-regularized Linear Regression via Stochastic Gradient Descent </strong>[ in Python]: Train a 2-layer neural network (i.e., linear regression) for age regression using the same data as in homework 1. Your prediction model should be ˆ<em>y </em>= <strong>x</strong><sup>&gt;</sup><strong>w</strong>+<em>b</em>. You should regularize <strong>w </strong>but not <em>b</em>. Note that, in contrast to Homework 1, this model includes a bias term.</li>

</ol>

Instead of optimizing the weights of the network with the closed formula, use stochastic gradient descent (SGD). There are several different hyperparameters that you will need to choose:

<ul>

 <li>Mini-batch size ˜<em>n</em>.</li>

 <li>Learning rate .</li>

 <li>Number of epochs.</li>

 <li><em>L</em><sub>2 </sub>Regularization strength <em>α</em>.</li>

</ul>

In order not to cheat (in the machine learning sense) – and thus overestimate the performance of the network – it is crucial to optimize the hyperparameters <strong>only </strong>on a <em>validation set</em>. (The training set would also be acceptable but typically leads to worse performance.) To create a validation set, simply set aside a fraction (e.g., 20%) of the age regression Xtr.npy and age regression ytr.npy to be the validation set; the remainder (80%) of these data files will constitute the “actual” training data. While there are fancier strategies (e.g., Bayesian optimization – another probabilistic method, by the way!) that can be used for hyperparameter optimization, it’s common to just use a grid search over a few values for each hyperparameter. In this problem, you are required to explore systematically (e.g., using nested for loops) at least 4 different parameters for each hyperparameter.

<strong>Performance evaluation</strong>: Once you have tuned the hyperparameters and optimized the weights so as to minimize the cost on the validation set, then: (1) <strong>stop </strong>training the network and (2) evaluate the network on the <strong>test </strong>set. Report the performance in terms of <em>unregularized </em>MSE.

<ol start="3">

 <li><strong>Regularization to encourage symmetry </strong>[10 points, on paper]: Faces (and some other kinds of data) tend to be left-right symmetric. How can you use <em>L</em><sub>2 </sub>regularization to discourage the weights from becoming too <em>a</em>symmetric? For simplicity, consider the case of a tiny 1×2 “image”. Hint: instead of using<strong>Iw </strong>as the <em>L</em><sub>2 </sub>penalty term (where <em>α </em>is the regularization strength), consider a different matrix in the middle. Your answer should consist of a 2×2 matrix <strong>S </strong>as well as an explanation of why it works.</li>

 <li><strong>Recursive state estimation in Hidden Markov Models </strong>[10 points, on paper]: Teachers try to monitor their student’s knowledge of the subject-matter, but teachers cannot directly peer inside students’ brains. Hence, they must make <em>inferences </em>about what the student knows based on students’ <em>observable behavior</em>, i.e., how they perform on tests, their facial expressions during class, etc. Let random variable (RV) <em>X<sub>t </sub></em>represent the student’s <em>state</em>, and let RV <em>Y<sub>t </sub></em>represent the student’s observable behavior, at time <em>t</em>. We can model the student as a Hidden Markov Model (HMM):

  <ul>

   <li><em>X<sub>t </sub></em>depends <em>only </em>on the previous state <em>X<sub>t</sub></em><sub>−1</sub>, <em>not </em>on any states prior to that (<em>Markov </em>property), i.e.</li>

  </ul></li>

</ol>

<em>P</em>(<em>x<sub>t </sub></em>| <em>x</em><sub>1</sub><em>,…,x<sub>t</sub></em><sub>−1</sub>) = <em>P</em>(<em>x<sub>t </sub></em>| <em>x<sub>t</sub></em><sub>−1</sub>)

<ul>

 <li>The student’s behavior <em>Y<sub>t </sub></em>depends only on his/her current state <em>X<sub>t</sub></em>, i.e.:</li>

</ul>

<em>P</em>(<em>y<sub>t </sub></em>| <em>x<sub>t</sub>,y</em><sub>1</sub><em>,…,y<sub>t</sub></em><sub>−1</sub>) = <em>P</em>(<em>y<sub>t </sub></em>| <em>x<sub>t</sub></em>)

<ul>

 <li><em>X<sub>t </sub></em>cannot be observed directly (it is <em>hidden</em>).</li>

</ul>

A probabilistic graphical model for the HMM is shown below, where only the observed RVs are shaded (the latent ones are transparent):

Suppose that the teacher already knows:

<ul>

 <li><em>P</em>(<em>y<sub>t </sub></em>| <em>x<sub>t</sub></em>) (<em>observation likelihood</em>), i.e., the probability distribution of the student’s behaviors given the student’s state.</li>

 <li><em>P</em>(<em>x<sub>t </sub></em>| <em>x<sub>t</sub></em><sub>−1</sub>) (<em>transition dynamics</em>), i.e., the probability distribution of the student’s current state given the student’s previous state.</li>

</ul>

The goal of the teacher is to estimate the student’s current state <em>X<sub>t </sub></em>given the <em>entire </em>history of observations <em>Y</em><sub>1</sub><em>,…,Y<sub>t </sub></em>he/she has made so far. Show that the teacher can, at each time <em>t</em>, update his/her belief <em>recursively</em>:

<em>P</em>(<em>x<sub>t </sub></em>| <em>y</em><sub>1</sub><em>,…,y<sub>t</sub></em>) ∝ <em>P</em>(<em>y<sub>t </sub></em>| <em>x<sub>t</sub></em>) <sup>X </sup><em>P</em>(<em>x<sub>t </sub></em>| <em>x<sub>t</sub></em>−<sub>1</sub>)<em>P</em>(<em>x<sub>t</sub></em>−<sub>1 </sub>| <em>y</em><sub>1</sub><em>,…,y<sub>t</sub></em>−<sub>1</sub>)

<em>x</em><em>t</em>−1

where <em>P</em>(<em>x<sub>t</sub></em><sub>−1 </sub>| <em>y</em><sub>1</sub><em>,…,y<sub>t</sub></em><sub>−1</sub>) is the teacher’s belief of the student’s state from time <em>t </em>− 1, and the summation is over every possible value of the previous state <em>x<sub>t</sub></em><sub>−1</sub>. <strong>Hint</strong>: You will need to use Bayes’ rule, i.e., for any RVs <em>A</em>, <em>B</em>, and <em>C</em>:

However, since the denominator in the right-hand side does not depend on <em>a</em>, this can also be rewritten as:

<em>P</em>(<em>a </em>| <em>b,c</em>) ∝ <em>P</em>(<em>b </em>| <em>a,c</em>)<em>P</em>(<em>a </em>| <em>c</em>)

<ol start="5">

 <li><strong>Linear-Gaussian prediction model </strong>[15 points, on paper]:</li>

</ol>

Probabilistic prediction models enable us to estimate not just the “most likely” or “expected” value of the target <em>y </em>(see figure above, right), but rather an entire <em>probability distribution </em>about which target values are more likely than others, given input <strong>x </strong>(see figure above, left). In particular, a linearGaussian model is a Gaussian distribution whose expected value (mean <em>µ</em>) is a linear function of the input features <strong>x</strong>, and whose variance is <em>σ</em><sup>2</sup>:

Note that, in general, <em>σ</em><sup>2 </sup>can also be a function of <strong>x </strong>(heteroscedastic case). Moreover, <em>non</em>-linear Gaussian models are also completely possible, e.g., the mean (and possibly the variance) of the Gaussian distribution is output by a deep neural network. However, in this problem, we will assume that <em>µ </em>is linear in <strong>x</strong>, and that <em>σ</em><sup>2 </sup>is the same for all <strong>x </strong>(homoscedastic case).

<strong>MLE</strong>: The parameters of probabilistic models are commonly optimized by <em>maximum likelihood estimation </em>(MLE). (Another common approach is <em>maximum a posteriori </em>estimation, which allows the practitioner to incorporate a “prior belief” about the parameters’ values.) Suppose the training dataset

. Let the parameters/weights of the linear-Gaussian model be <strong>w</strong>, such that the

mean <em>µ </em>= <strong>x</strong><sup>&gt;</sup><strong>w</strong>. Prove that the MLE of <strong>w </strong>and <em>σ</em><sup>2 </sup>given D is:

!

<strong>w</strong>

Note that this solution – derived based on <em>maximizing </em>probability – is exactly the same as the optimal weights of a 2-layer neural network optimized to <em>minimize </em>MSE.

Hint: Follow the same strategy as the MLE derivation for a biased coin in Class2.pdf. For a linearGaussian model, the argmax of the likelihood equals the argmax of the log-likelihood. The log of the Gaussian likelihood simplifies beautifully.

Put your code in a Python file called homework2 WPIUSERNAME1.py

(or homework2 WPIUSERNAME1 WPIUSERNAME2.py for teams). For the proofs, please create a PDF called homework2 WPIUSERNAME1.pdf

(or homework2 WPIUSERNAME1 WPIUSERNAME2.pdf for teams). Create a Zip file containing both your Python and PDF files, and then submit on Canvas.