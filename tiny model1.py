input_value = 2.0
correct_answer = 4.0
weight = 0.3
learning_rate = 0.1

for step in range(30):
    prediction = input_value * weight
    error = prediction - correct_answer

    print(step, "weight:", weight, "Prediction:", prediction)

    gradient = error * input_value

    weight = weight - learning_rate * gradient