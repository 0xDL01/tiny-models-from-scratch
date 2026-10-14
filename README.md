# Tiny Models From Scratch

This project explores how models learn by building small experiments in Python. The aim is to understand each calculation before moving to larger networks.

## Experiment 1: Learning One Weight

The first experiment, `tiny_model1.py`, uses one input and one adjustable weight. Its task is to produce an output of 4 from an input of 2.

The prediction is calculated as:

```python
prediction = input_value * weight
```

With a starting weight of 0.3, the first prediction is 0.6.

## Learning Process

The model compares its prediction with the correct answer, then updates the weight:

```python
error = prediction - correct_answer
gradient = error * input_value
weight = weight - learning_rate * gradient
```

The gradient is derived from the loss function:

```text
loss = 0.5 × error²
```

It describes how the loss changes with the weight. The learning rate controls the size of each update.

For the first iteration:

```text
Prediction = 2 × 0.3 = 0.6
Error = 0.6 − 4 = −3.4
Gradient = −3.4 × 2 = −6.8
Updated weight = 0.3 − 0.1 × (−6.8) = 0.98
```

The next prediction is 1.96, which is closer to the target.

## Implementation

```python
input_value = 2.0
correct_answer = 4.0
weight = 0.3
learning_rate = 0.1

for step in range(20):
    prediction = input_value * weight
    error = prediction - correct_answer

    print(step, "Weight:", weight, "Prediction:", prediction)

    gradient = error * input_value
    weight = weight - learning_rate * gradient
```

## Running the Experiment

Python 3 is required. No additional packages are needed.

```bash
python3 tiny_model1.py
```

## Observations

Over 20 iterations, the weight approaches 2 and the prediction approaches 4. As the error decreases, the updates become smaller.

The target weight is not assigned directly. It is learned through repeated error-based adjustments. Each printed row shows the values before that iteration’s update.

This is supervised learning because the correct answer is supplied. The model is a simple linear model without a bias or nonlinear activation.

## Limitations and Next Step

This experiment uses only one training example. It demonstrates how a parameter changes during learning, rather than the ability to learn complex patterns.

The learned weight is not saved and resets when the script restarts.

The next experiment will introduce a bias and multiple training examples, allowing us to examine how two parameters are learned together.
