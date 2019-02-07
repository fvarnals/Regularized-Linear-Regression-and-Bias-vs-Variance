function [avg_error_train, avg_error_val] = ...
  avg_learningCurve(X, y, Xval, yval, lambda)

m = size(X,1);

random_iterations = 50

avg_error_train = zeros(m,1);
avg_error_val = zeros(m, 1);

error_train_matrix = zeros(m, random_iterations);
error_val_matrix = zeros(m, random_iterations);

theta = [1; 1];

for i = 1:m
  for iteration = 1:random_iterations
    test_sel = randperm(m, i);
    X_test = X(test_sel, :);
    y_test = y(test_sel);
    thetas = trainLinearReg(X_test, y_test, lambda);

    val_sel = randperm(m, i);
    X_val = Xval(val_sel, :);
    y_val = yval(val_sel);

    error_train(i, iteration) = linearRegCostFunction(X_test, y_test, thetas, 0);

    error_val(i, iteration) = linearRegCostFunction(X_val, y_val, thetas, 0);
  end
end

avg_error_train = mean(error_train, 2);
avg_error_val = mean(error_val, 2);
