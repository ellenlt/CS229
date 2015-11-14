#!/usr/local/bin/octave

[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');
trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.

% YOUR CODE HERE

% Find row indices of emails classified as spam or non-spam
spamIndices = find(trainCategory==1);
nonSpamIndices = find(trainCategory==0);

% Generate matrices containing spam and non-spam emails
spamMatrix = trainMatrix(spamIndices,:);
nonSpamMatrix = trainMatrix(nonSpamIndices,:);

% Vector of token counts for how many times
% each token appears in spam and non-spam
spamCounts = sum(spamMatrix);
nonSpamCounts = sum(nonSpamMatrix);

% logSpamCPs, logNonSpamCPs
% are (1 x numTokens) vectors containing log conditional probabilities
% of seeing token i in an email,
% given the email is a spam or non-spam. With LaPlace smoothing.
% Computed with with LaPlace smoothing and using following formula:
% For spamProbs:
	% Numerator = # times token i appears in spam + 1
	% Denominator = # tokens total in spam + # tokens in vocab
logSpamCPs = log((spamCounts.+1)./(sum(spamCounts)+numTokens));
logNonSpamCPs = log((nonSpamCounts.+1)./(sum(nonSpamCounts)+numTokens));

% Log probabilities (scalars) that an email is spam or not spam
logPSpam = log(rows(spamMatrix)/numTrainDocs);
logPNonSpam = log(rows(nonSpamMatrix)/numTrainDocs);

%------------------------------------------------------------
% Code to print out top most indicative words

% (1 x numTokens) vector containing correlation measures for each word.
% The more positive the value, the more indicative the word of spam
indicative = logSpamCPs.-logNonSpamCPs;

[sortedIndicative, sortIndices] = sort(indicative, 'descend');
maxValues = sortedIndicative(1:5);
maxValueIndices = sortIndices(1:5);

tokenMatrix = strsplit(tokenlist);
disp("")
disp("Top 5 tokens most indicative of spam:")
tokenMatrix{1,maxValueIndices}
disp("")

