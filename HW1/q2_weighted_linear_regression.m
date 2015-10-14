% Implements weighted linear regression using normal equations
% q2x.dat.txt contains inputs x in R^2
% q2y.dat.txt contains outputs y in {0,1}

input_filename = "q2x.dat.txt";
output_filename = "q2y.dat.txt";

% Load data
X = importdata(input_filename);
y = importdata(output_filename);

% Performs weighted linear regression method and plots the fitted curve
function plotWeightedLinRegression(X,y)
	spacing = exp(-3);	% Amount of space in between each point
	% Query vector containing all the x values
	% for which you want to predict y, added vertically
	queryX = (linspace(min(X), max(X), abs(max(X)-min(X))/spacing))';

	% Append col of ones for intercept term
	queryX = [ones(size(queryX,1),1), queryX];
	X = [ones(size(X,1),1), X];
	[m,n] = size(queryX);

	for i = 1:m
		queryXi=queryX(i,:)';
		W = calculateWeightMatrix(queryXi, X);
		theta = (X'*W*X)\(X'*W*y);
		hx = theta'*queryXi;
		plot(queryXi(2), hx, 'b.'); hold on
	end

end

% Calculates the weight matrix for a particular query point xi
function W = calculateWeightMatrix(xi, X)
	tau = 0.8;	% Bandwidth parameter
	[m,n] = size(X);
	W = zeros(m,m);

	for i = 1:m
		delta = xi-X(i,:)';
		W(i,i) = exp(-delta'*delta/(2*tau^2));
	end
end

% Plots data on axes x and y along with the
% fitted linear regression line
function plotTrainingData(X,y,theta)
	[m,n] = size(X);

	% X has one feature
	plot(X, y, 'ro'); hold on
end

% Sigmoid function
function a = sigmoid(z)
	a = 1.0 ./ (1.0 + exp(-z));
end


plotWeightedLinRegression(X,y)
plotTrainingData(X,y)
pause