% Implements weighted linear regression using normal equations
% q2x.dat.txt contains inputs x in R^2
% q2y.dat.txt contains outputs y in {0,1}

input_filename = "q2x.dat.txt";
output_filename = "q2y.dat.txt";

% Load data
X = importdata(input_filename);
y = importdata(output_filename);

% Performs weighted linear regression method and plots the fitted curve
function [queryX, hx1, hx2, hx3, hx4] = plotWeightedLinRegression(X,y)
	spacing = exp(-1);	% Amount of space in between each point
	% Query vector containing all the x values
	% for which you want to predict y, added vertically
	queryX = (linspace(min(X), max(X), abs(max(X)-min(X))/spacing))';

	% Append col of ones for intercept term
	queryX = [ones(size(queryX,1),1), queryX];
	X = [ones(size(X,1),1), X];
	[m,n] = size(queryX);

	% Vector containing predicted outputs
	hx1 = zeros(m,1);

	for i = 1:m
		queryXi=queryX(i,:)';
		W = calculateWeightMatrix(queryXi, X, 0.1);
		theta = (X'*W*X)\(X'*W*y);
		hx1(i) = theta'*queryXi;
	end

	hx2 = zeros(m,1);

	for i = 1:m
		queryXi=queryX(i,:)';
		W = calculateWeightMatrix(queryXi, X, 0.3);
		theta = (X'*W*X)\(X'*W*y);
		hx2(i) = theta'*queryXi;
	end

	hx3 = zeros(m,1);

	for i = 1:m
		queryXi=queryX(i,:)';
		W = calculateWeightMatrix(queryXi, X, 2);
		theta = (X'*W*X)\(X'*W*y);
		hx3(i) = theta'*queryXi;
	end

	hx4 = zeros(m,1);

	for i = 1:m
		queryXi=queryX(i,:)';
		W = calculateWeightMatrix(queryXi, X, 10);
		theta = (X'*W*X)\(X'*W*y);
		hx4(i) = theta'*queryXi;
	end
end

% Calculates the weight matrix for a particular query point xi
function W = calculateWeightMatrix(xi, X, tau)
	[m,n] = size(X);
	W = zeros(m,m);

	for i = 1:m
		delta = xi-X(i,:)';
		W(i,i) = exp(-delta'*delta/(2*tau^2));
	end
end

% Plots data and predictions
function plotData(X,y, queryX, hx1, hx2, hx3, hx4)
	plot(X, y, 'r+'); hold on
	plot(queryX(:,2), hx1, 'b', 'LineWidth',2); hold on
	plot(queryX(:,2), hx2, 'g', 'LineWidth',2); hold on
	plot(queryX(:,2), hx3, 'c', 'LineWidth',2); hold on
	plot(queryX(:,2), hx4, 'm', 'LineWidth',2);

	legend('Predictions','tau = 0.1', 'tau = 0.3', 'tau = 2', 'tau = 10', 'Location','southeast');
	ylabel('y'); xlabel('x');
	title('Weighted Linear Regression');
end

% Sigmoid function
function a = sigmoid(z)
	a = 1.0 ./ (1.0 + exp(-z));
end


[queryX, hx1, hx2, hx3, hx4] = plotWeightedLinRegression(X,y);
plotData(X,y, queryX, hx1, hx2, hx3, hx4)
pause