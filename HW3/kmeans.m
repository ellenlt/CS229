pkg load statistics

%r = rand(5,5);
%g = rand(5,5);
%b = rand(5,5);
%B(:,:,1) = r;
%B(:,:,2) = g;
%B(:,:,3) = b;

% compressImage
% Input: imageData: 3D matrix where m(:,:,1), m(:,:,2), and m(:,:,3) are nxn 2D matrices
%		 			containing the R,G,B values for each of n^2 pixels
% 		  numClusters: the initial number of centroids
%		 minIters: minimum iterations to run k means
% Output: 
function compressedImage = compressImage(imageData, numClusters, minIters)
	data = flattenMatrix(imageData);
	[clusterAssignments, centroids] = runKMeans(data, minIters, initializeClusters(data, numClusters), numClusters);
	
	% nxn matrix containing the cluster assignments (1,...,numClusters) for each pixel
	clusterAssignments = reshape(clusterAssignments, size(imageData(:,:,1)));
	
	rVals = reshape(centroids(clusterAssignments, 1), size(clusterAssignments))
	gVals = reshape(centroids(clusterAssignments, 2), size(clusterAssignments))
	bVals = reshape(centroids(clusterAssignments, 3), size(clusterAssignments))

	compressedImage(:,:,1) = rVals;
	compressedImage(:,:,2) = gVals;
	compressedImage(:,:,3) = bVals;
end

% runKMeans
% Input: data: is a mxn matrix with m samples as rows and n features as columns
%		 iters: minimum iterations to run k means
% 		 centroids: kxn matrix with centroids as rows
% 		  numClusters: the initial number of centroids
% Output: clusterAssignments: mx1 list of the cluster assignments (1,...,numClusters) for each sample
% 		  centroids: kxn matrix with centroids as rows
function [clusterAssignments, centroids] = runKMeans(data, minIters, centroids, numClusters)
	clusterAssignments = zeros(length(data),1);
	i = 0;
	while true
		% Assign each point to closest cluster
		newClusterAssignments = assignClusters(data, centroids);
        % Stop if cluster assignments converge, for at least 30 iterations
        if newClusterAssignments == clusterAssignments && i > minIters
            clusterAssignments = newClusterAssignments;
            break;
        end
        clusterAssignments = newClusterAssignments;
        % Recompute centroids to be the average of the data points assigned to them
        centroids = updateCentroids(clusterAssignments, centroids, data, numClusters);
        i = i+1;
  	end
 end  

% assignClusters
% Assigns each sample to a cluster based on closest Euclidean distance
% Input: data: mxn matrix with m samples as rows and n conditions/features as columns
% 		 centroids: kxn matrix with centroids as rows
% Output: mx1 list of indices indicating which centroids each sample has been assigned to
function clusterAssignments = assignClusters(data, centroids)
    % mxk matrix where the (m,k)th element is the distance between
    % the mth sample and the kth centroid
    minDistances = pdist2(data, centroids);
    % Find the index of the centroid closest to each row/sample
    [val,clusterAssignments] = min(minDistances,[],2);
end

% updateCentroids
% Updates centroid locations to be the average of data points in a cluster.
% Keeps old centroid locations for empty clusters.
% Inputs: clusterAssignments: mx1 vector with the assigned cluster numbers for each sample
% 		  centroids: kxn matrix with centroids as rows
%         data: mxn matrix with m samples as rows and n conditions/features as columns
% 		  numClusters: the initial number of centroids
% Output: kxn vector of updated centroids
function updatedCentroids = updateCentroids(clusterAssignments, centroids, data, numClusters)
	clusters = unique(clusterAssignments);
	updatedCentroids = zeros(numClusters,3);
	for i=1:numClusters
	    % If a centroid has no points assigned to it, don't update it
	    if ismember(i,clusters) == 0
	    	updatedCentroids(i,:) = centroids(i,:);
	    else
		    updatedCentroids(i,:) = mean(data(clusterAssignments==i,:));
		end
	end
end

% initializeClusters
% Input: data with m samples as rows and n conditions/features as columns
% Output: kxn matrix with n centroids as rows, randomly chosen from data
function centroids = initializeClusters(data, k)
    centroids = data(randperm(length(data), k),:);
end

% flattenMatrix
% Input: 3D matrix where m(:,:,1), m(:,:,2), and m(:,:,3) are all nxn 2D matrices
% Ouput: 2D matrix of nx3 dimension, where each row represents a point in the starting nxn matrices
%		 and the 3 columns contain the values from the 3 initial matrices corresponding to that point
function data = flattenMatrix(m)
	data = [reshape(m(:,:,1),numel(m(:,:,1)),1),reshape(m(:,:,2),numel(m(:,:,2)),1),reshape(m(:,:,3),numel(m(:,:,3)),1)];
end

A = double(imread('mandrill-large.tiff'));
imshow(uint8(round(A)));

B = double(imread('mandrill-small.tiff'));
imshow(uint8(round(B)));

compressedImage = compressImage(B, 16, 30);
imshow(uint8(round(compressedImage)));

