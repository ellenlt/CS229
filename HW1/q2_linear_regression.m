% Implements unweighted linear regression using normal equations
% q2x.dat.txt contains inputs x in R^2
% q2y.dat.txt contains outputs y in {0,1}

input_filename = "q2x.dat.txt";
output_filename = "q2y.dat.txt";

% Load data
X = importdata(input_filename);
y = importdata(output_filename);

% Performs unweighted linear regression method and returns the parameters theta
function theta = linRegression(X,y)
	X = [ones(size(X,1),1), X]; % Append col of ones for intercept term

	theta = X'*X\X'*y;
end

% Plots data on axes x and y along with the
% fitted linear regression line
function plotTrainingData(X,y,theta)
	[m,n] = size(X);

	% X has one feature
	plot(X, y, 'bo'); hold on

	% Plot linear regression line using theta
	a = linspace(min(X), max(X));
	b = theta(1)+theta(2)*a;
	plot(a,b, 'k','LineWidth',2);
	
	legend('Data','Regression line')
	xlabel('x'); ylabel('y');
	title('Unweighted Linear Regression');
end

% Sigmoid function
function a = sigmoid(z)
	a = 1.0 ./ (1.0 + exp(-z));
end

theta = linRegression(X,y)
plotTrainingData(X,y,theta)
pause