% An artificial neural network consists of a network of neurons, each with
% an activation function and a set of input and output neurons. The number
% of neurons in the input layer will always be equal to the number of
% features in the training set. We apply the values of each feature of an
% observation to a neuron in the input layer. The activation function
% determines whether a neuron will fire based on its inputs. Each neuron
% has a weight associated with each input. These weights might initially be
% random.
% The output of every neuron feeds into the neurons of the next layer. The
% signal propagates through the network in this way until it reaches the
% output layer. The value is then compared to the expected value, and the
% error is determined.
% To minimize the error, we will back propagate through the network,
% subtracting the derivative of error with respect to input weight from 
% each weight.
% This process trains the neural network. After the network is trained,
% data from outside the training set can forward propagate through the
% network and produce a result with minimal error.

clear all
data = csvread('fetal.csv');

learning_rate_start = 0.01;
learning_rate_factor = 0.9;

% Shuffle all rows
data = data(randperm(length(data)), :);

% Feature scaling is required to achieve reasonable results.
for n = 1:21
    data(:,n) = data(:,n)/norm(data(:,n));
end    

data = [ones(2126, 1), data];

%data = sortrows(data, 23);

% The last column is the fetal state classification
input_data = data(:,1:22);
output_data = data(:,end);

% Treat a suspect classification as Pathologic
output_data(output_data == 1) = 0;
output_data(output_data == 2) = 1;
output_data(output_data == 3) = 1;

% Set the number of observations to make up the training set
% All others will be used for testing
number_to_train = 1000;

train = input_data(1:number_to_train,:);% an activation function and a set of input and output neurons. The number

train_output = output_data(1:number_to_train,:);
test = input_data((number_to_train+1):end,:);

% The neural network learning rate. If this value is too high, the neural
% net may diverge.
eta = learning_rate_start;

n_rows = length(train);

% The number of layers and artificial neurons per layer. The size of the
% input layer must be equal to the number of features per observation (the
% number of columns, excluding the classification)
neural_net = PERCEPTRON([22 5 1]);

num_pos = 0;
num_neg = 0;
for j = 1:number_to_train
    if output_data(j) == 1
        num_pos = num_pos + 1;
    else
        num_neg = num_neg + 1;
    end
end

num_pos
num_neg

t1 = datetime('now');
loop = 1;
% Apply the backpropagation algorithm to each input vector
while loop
    t1 = datetime('now');
    loop = 0;
    fprintf('Starting backprop\n');
    eta = learning_rate_start;
    neural_net = PERCEPTRON([22 5 1]);
    for k = 1:300
        eta = eta*learning_rate_factor;
        for j = 1:number_to_train
            trainslice = train(j,:)';

            train_output_vector = train_output(j);
            neural_net.backprop(trainslice, train_output_vector', eta);
            if neural_net.divergence
                loop = 1;
                fprintf('Neural network has diverged\n');
                break
            end
        end
        if neural_net.divergence
            fprintf('Aborting\n');
            dt = between(t1, datetime('now'))
            break
        end
    end
end

if ~neural_net.divergence
    fprintf('Success!\n');
    dt = between(t1, datetime('now'))
end