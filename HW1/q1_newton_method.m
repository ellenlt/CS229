% Implements Newtons method for optimizing l(theta) and 
% applies it to fit a logistic regression model to the data.
% q1x.dat.txt contains inputs x in R^2
% q1y.dat.txt contains outputs y in {0,1}

input_filename = "q1x.dat.txt";
output_filename = "q1y.dat.txt";

% Load data
X = importdata(input_filename);
y = importdata(output_filename);

% Performs Newton-Raphson method and returns the parameters theta
function theta = newtonsMethod(X,y)
	max_iters =300;
	threshold = exp(-6);	% Threshold for change in thetas

	X = [ones(size(X,1),1), X]; % Append col of ones for intercept term
	[m,n] = size(X);	% Size of X including intercept term
	theta = zeros(n, 1);	% Initialize theta

	for k = 1:max_iters
		for i = 1:m
			new_theta = theta - computeHessian(X,theta)\computeGradient(X,y,theta);
			if abs(new_theta - theta) < threshold return;
			end
			theta = new_theta;
		end
	end
end

% Computes the Hessian and returns as an nxn matrix
function hessian = computeHessian(X,theta)
	% i of m total instances, j of n total features
	[m,n] = size(X);
	hessian = zeros(n,n);
	for j = 1:n
		for k = 1:n
			% Compute the hypothesis
			hx = sigmoid(X*theta);
			% Updates the hessian using this training instance
			Hjk = sum(-X(:,j).*X(:,k).*hx.*(1-hx));
			hessian(j,k) = Hjk;
			hessian(k,j) = Hjk;
		end
	end
end

% Computes the gradient and returns as a nx1 matrix
function grad = computeGradient(X,y,theta)
	[m,n] = size(X);
	grad = zeros(n,1);
	hx = sigmoid(X*theta);
	grad = X'*(y-hx);
end

% Plots data on axes x1, x2 with different symbols for data points where
% y = 0 or y = 1. Also plots logistic regression decision boundary line
function plotTrainingData(X,y,theta)
	[m,n] = size(X);
	% Find returns the indices of the rows meeting the specified condition
	pos = find(y == 1); neg = find(y == 0);

	% The features are in the 1st and 2nd columns of X
	plot(X(pos, 1), X(pos,2), 'rx'); hold on
	plot(X(neg, 1), X(neg, 2), 'bo'); hold on

	% Decision boundary includes values of x such that hx = 0.5
	a = linspace(min(X(:,1)), max(X(:,1)));
	b = (0.5 - theta(1) - theta(2)*a)/theta(3);
	plot(a,b, 'k','LineWidth',2);

	legend('y=1','y=0')

	pause
end

% Sigmoid function
function a = sigmoid(z)
	a = 1.0 ./ (1.0 + exp(-z));
end

theta = newtonsMethod(X,y)
plotTrainingData(X,y,theta)

