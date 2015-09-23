function in = vl_nntt_backward(layer, in, out)
% VL_NNTT_BACKWARD  Tensor Train layer backward pass
%    in = VL_NNTT_BACKWARD(layer, in, out)  computes all the necessary
%    derivatives for the back-propagation algorithm.
%
%    The transformation of the layer (the forward pass) is defined as:
%       Y = out.x = layer.W * in.x + biases,
%    where biases are stored in layer.weights{2}.
%
%    in.dzdx is the derivative of the neural network's out Z with respect to
%       the input in.x;
%    layer.dzdw{1} is the derivative of Z w.r.t. the cores of the
%       TT-decomposition of the matrix W;
%    layer.dzdw{2} is the derivative of Z w.r.t. the biases.
%
%    in.x is of size inHeight x inWidth x inChannels x batchSize.
%
%    The complexity of the backward pass is
%       O(ttRank^4 * modeSize * numTTCores^2 * inHeight * inWidth * inChannels * batchSize),
%    where
%       inHeight * inWidth * inChannels == modeSize^numTTCores.

layer.W.core = layer.weights{1};
W = layer.W;
[inHeight, inWidth, inChannels, batchSize] = size(in.x);

in.dzdx = full(W' * reshape(out.dzdx, [], batchSize));
in.dzdx = reshape(in.dzdx, inHeight, inWidth, inChannels, batchSize);

if numel(layer.weights{2}) > 0
    layer.dzdw{2} = sum(out.dzdx, 4);
else
    layer.dzdw{2} = [];
end
DZDWCore = zeros(size(W.core), 'single');
if isa(in.x, 'gpuArray')
    DZDWCore = gpuArray(DZDWCore);
end
rankArr = rank(W);
corePos = W.ps;
% We have a TT matrix W(i1, ..., in; j1, ..., jd).
% Y = sum_{j,imageIdx} W(i, j) * in.x(j, imageIdx) + b(j).
numDims = length(W.n);
coreArr = core2cell(W);
% On the beginning of the derDim iteration rightSum depends on:
% rightSum(alpha_derDim+2, i_derDim+2, ..., i_n, imageIdx, j_1, j_2, ..., j_derDim+1).
rightSum = reshape(in.x, [prod(W.m), batchSize]);
rightSum = rightSum';
for derDim = numDims:-1:1
    % Computing the derivative of the Y w.r.t. the G_{derDim}.
    if (derDim < numDims)
        rightDim = derDim + 1;
        sumSize = W.m(rightDim) * rankArr(rightDim+1);
        core = reshape(coreArr{rightDim}, [], sumSize);
        rightSum = reshape(rightSum, [], W.m(rightDim));
        rightSum = core * reshape(rightSum', sumSize, []);
    end

    if derDim >= 2
        % Permute core dimensions from
        % alpha_derDim-1, i_derDim-1, j_derDim-1, alpha_derDim
        % to
        % alpha_derDim-1, i_derDim-1, alpha_derDim, j_derDim-1.
        core = permute(coreArr{derDim-1}, [1, 2, 4, 3]);
        core = reshape(core, [], W.m(derDim-1));
        % Permute (shift) dimensions from
        % alpha_derDim+1, i_derDim+1, ..., i_n, imageIdx, j_1, j_2, ..., j_derDim
        % to
        % j_derDim-1, j_derDim, alpha_derDim+1, i_derDim+1, ..., i_n, imageIdx, j_1, j_2, ..., j_derDim-2.
        leftSum = reshape(rightSum, [rankArr(derDim+1)*prod(W.n(derDim+1:end))*batchSize*prod(W.m(1:derDim-2)), prod(W.m(derDim-1:derDim))]);
        leftSum = core * reshape(leftSum.', W.m(derDim-1), []);
        % Permute dimensions from
        % alpha_derDim-1, i_derDim-1, alpha_derDim, j_derDim, alpha_derDim+1,
        %       j_1, j_2, ..., j_derDim-2, i_derDim+1, ..., i_n, imageIdx
        % to
        % alpha_derDim-1, i_derDim-1, i_derDim+1, ..., i_n,
        %       imageIdx, alpha_derDim, j_derDim, alpha_derDim+1,
        %       j_1, ..., j_derDim-2.
        leftSumDims = [rankArr(derDim-1)*W.n(derDim-1), rankArr(derDim)*W.m(derDim)*rankArr(derDim+1), ...
                       prod(W.n(derDim+1:end))*batchSize, prod(W.m(1:derDim-2))];
        leftSum = reshape(leftSum, leftSumDims);
        leftSum = permute(leftSum, [1, 3, 2, 4]);
        %
        % On the beginning of the leftDim iteration leftSum depends on:
        % leftSum(alpha_leftDim+1,
        %       i_leftDim+1, ..., i_derDim-1, i_derDim+1, ..., i_n,
        %       imageIdx,
        %       alpha_derDim, j_derDim, alpha_derDim+1)
        for leftDim = derDim-2:-1:1
            sumSize = W.m(leftDim) * rankArr(leftDim+1);
            core = reshape(coreArr{leftDim}, [], sumSize);
            leftSum = reshape(leftSum, [], W.m(leftDim));
            leftSum = core * reshape(leftSum', sumSize, []);
        end
    elseif (derDim == 1)
        % Permute (shift) dimensions from
        % alpha_2, i_2, ..., i_n, imageIdx, j_1
        % to
        % i_2, ..., i_n, imageIdx, j_1, alpha_2
        leftSum = reshape(rightSum, rankArr(derDim+1), [], batchSize, W.m(derDim));
        leftSum = permute(leftSum, [2, 3, 4, 1]);
    else
        error('Something bad happened(');
    end
    coreSize = rankArr(derDim) * W.n(derDim) * W.m(derDim) * rankArr(derDim+1);
    leftISize = prod(W.n(1:derDim-1));
    rightISize = prod(W.n(derDim+1:end));
    % Permute dimensions from
    % i_1, ..., i_n, imageIdx
    % to
    % i_derDim, i_1, ..., i_derDim-1, i_derDim+1, ..., i_n, imageIdx
    currout.dzdx = reshape(out.dzdx, leftISize, W.n(derDim), rightISize*batchSize);
    currout.dzdx = permute(currout.dzdx, [2, 1, 3]);
    sumSize = leftISize * rightISize * batchSize;
    der = reshape(currout.dzdx, [], sumSize) * reshape(leftSum, sumSize, []);

    % Permute derivative dimensions from
    % i_derDim, alpha_derDim, j_derDim, alpha_derDim+1
    % to
    % alpha_derDim, i_derDim, j_derDim, alpha_derDim+1.
    der = reshape(der, W.n(derDim), rankArr(derDim), W.m(derDim)*rankArr(derDim+1));
    der = permute(der, [2, 1, 3]);
    DZDWCore(corePos(derDim):corePos(derDim+1)-1) = der;
end
layer.dzdw{1} = DZDWCore;
end
