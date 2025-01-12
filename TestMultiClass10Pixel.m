clear all;
rng(3); % For reproducibility

% Initialize the input data for 10x10 pixel images. 
X = zeros(10, 10, 5);

% Fill in the 10x10 matrices with example data (simplified patterns).
% These are NOT actual handwritten digits but serve as placeholders.
% Each slice of the 3D matrix 'X' corresponds to one "image."

% Image 1
X(:, :, 1) = [
    0 1 1 1 1 1 1 1 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 1 1 1 1 1 1 1 0;
];

% Image 2
X(:, :, 2) = [
    0 0 0 1 1 1 1 0 0 0;
    0 0 1 1 0 0 1 1 0 0;
    0 1 1 0 0 0 0 1 1 0;
    0 1 1 0 0 0 0 1 1 0;
    0 0 0 0 0 0 0 1 1 0;
    0 0 0 0 0 0 1 1 0 0;
    0 0 0 0 0 1 1 0 0 0;
    0 0 0 0 1 1 0 0 0 0;
    0 0 0 1 1 0 0 0 0 0;
    0 1 1 1 1 1 1 1 1 0;
];

% Image 3
X(:, :, 3) = [
    0 1 1 1 1 1 1 1 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 1 1 1 1 1 1 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 1 1 1 1 1 1 1 0;
];

% Image 4
X(:, :, 4) = [
    0 0 0 1 1 1 1 0 0 0;
    0 0 1 1 0 0 1 1 0 0;
    0 1 1 0 0 0 0 1 1 0;
    0 1 1 0 0 0 0 1 1 0;
    0 0 0 0 0 0 0 1 1 0;
    0 0 0 0 0 0 0 1 1 0;
    0 0 0 0 0 0 0 1 1 0;
    0 1 1 0 0 0 0 1 1 0;
    0 1 1 0 0 0 0 1 1 0;
    0 0 1 1 1 1 1 1 0 0;
];

% Image 5
X(:, :, 5) = [
    0 1 1 1 1 1 1 1 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 1 1 1 1 1 1 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0 1 0;
    0 1 1 1 1 1 1 1 1 0;
];

% Your target matrices, one-hot encoded, assuming 5 different classes.
D = [ 
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1
];

% Initializing weights randomly
W1 = 2*rand(50, 100) - 1; % Since the input is now 10*10 pixels, we have 100 inputs.
W2 = 2*rand(5, 50) - 1;

% Assuming you have functions 'MultiClass', 'Sigmoid', and 'Softmax' defined elsewhere,
% you would proceed with training using 'MultiClass' and then perform inference.

for epoch = 1:10000 % Training epochs
    [W1, W2] = MultiClass10Pixel(W1, W2, X, D); % This function should update the weights W1 and W2.
end

N = 5; % Inference on 5 samples
for k = 1:N
    x = reshape(X(:, :, k), 100, 1); % Reshaping each image into a vector
    v1 = W1*x; % Forward pass
    y1 = Sigmoid(v1); % Activation function
    v = W2*y1; % Second layer
    y = Softmax(v); % Prediction
    disp(y); % Displaying the output
end
