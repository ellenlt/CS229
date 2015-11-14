nb_train;

[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);


% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE

% logSpamCPs, logNonSpamCPs
% are (1 x numTokens) vectors containing log conditional probabilities
% log P(token|spam) and log P(token|non-spam)
% Computed using LaPlace smoothing.

% logPSpam, logPNonSpam
% are scalars equal to the log probability that an email is spam or non-spam

% Repeat the rows of logSpamCPs and logNonSpamCPs to generate a
% (numTestDocs x numTokens) matrix
logSpamCPsExpanded = repmat(logSpamCPs, numTestDocs, 1);
logNonSpamCPsExpanded = repmat(logNonSpamCPs, numTestDocs, 1);

% Records positions in testMatrix where a word appears
wordIdx = testMatrix>0;

% aggregateSpamMatrix and aggregateNonSpamMatrix are (numTestDocs x numTokens) matrices
% containing n*log P(token|spam) and n*log P(token|non-spam) for words which appear n times.
aggregateSpamMatrix = zeros(numTestDocs, numTokens);
aggregateSpamMatrix(wordIdx) = logSpamCPsExpanded(wordIdx).*testMatrix(wordIdx);
aggregateNonSpamMatrix = zeros(numTestDocs, numTokens);
aggregateNonSpamMatrix(wordIdx) = logNonSpamCPsExpanded(wordIdx).*testMatrix(wordIdx);

% spam and nonSpam
% are (numTestDocs x 1) vectors that contain
% log(P(W|spam)P(spam)) and log(P(W|non-spam)P(non-spam))
% quantities for each document
spam = sum(aggregateSpamMatrix,2)+logPSpam;
nonSpam = sum(aggregateNonSpamMatrix,2)+logPNonSpam;

% If log(P(W|spam)P(spam)) >= log(P(W|non-spam)P(non-spam))
% label the email as spam
spamIdx = spam >= nonSpam;
output(spamIdx) = 1;


%---------------


% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (category(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
disp("Test classification error:")
error/numTestDocs


