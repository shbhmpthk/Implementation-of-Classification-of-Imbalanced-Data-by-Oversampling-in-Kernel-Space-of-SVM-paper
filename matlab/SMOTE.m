function [outFeat] = SMOTE(inFeat, N, k)
% Create numOut synthetic samples from obervations given as rows of
% in_features; assumes they are all the same class
% N is upsample factor (N * 100 = Percent SMOTE)
% k selects number of nearest neigbours to use
% See: Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling 
% technique." Journal of artificial intelligence research 16 (2002): 
% 321-357 (https://www.jair.org/media/953/live-953-2037-jair.pdf)
% TODO: Normalisation?
% TODO: Implement for <1 and non-integer values
% Adam Hartwell 2016

    if N < 1 && rem(N,1) == 0 
        error('Please use an oversampling integer multiple >=1'); 
    end
    
    numObs = size(inFeat, 1); % Number of observations (input examples)
    numVars = size(inFeat, 2); % Number of variables per observation
    
    
    outFeat = zeros(numObs*N, numVars); % Output N * the number of input examples
    
    IDX = knnsearch(inFeat, inFeat, 'K', k, 'Distance', 'euclidean');
    for ii = 1:numObs
        for jj = 1:N
            nn = randi(k,1); % Which of the nearest samples we'll pull towards
            nnValues = inFeat(IDX(ii, nn), :); 
            
            gap = rand(1,numVars);
            dif = nnValues - inFeat(ii,:);
            
            outIndex = (ii-1)*N + jj;
            outFeat(outIndex, :) = inFeat(ii,:) + dif.*gap;
        end
    end
end