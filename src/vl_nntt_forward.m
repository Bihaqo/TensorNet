function out = vl_nntt_forward(layer, in, out)
% VL_NNTT_FORWARD  Tensor Train layer forward pass
%    out = VL_NNTT_FORWARD(layer, in, out) applies a linear operator layer.W
%    which is represented in the TT-format to the data in.x:
%       out.x = layer.W * in.x + biases,
%    where biases are stored in layer.weights{2}.
%
%    The parameters of the model are the values of TT cores (layer.weights{1})
%    and the biases (layer.weights{2}).
%
%    in.x is of size inHeight x inWidth x inChannels x batchSize.
%
%    The complexity of the forward pass is
%       O(ttRank^2 * modeSize * numTTCores * inHeight * inWidth * inChannels * batchSize),
%    where
%       inHeight * inWidth * inChannels == modeSize^numTTCores.

layer.W.core = layer.weights{1};
[inHeight, inWidth, inChannels, batchSize] = size(in.x);


out.x = full(layer.W * reshape(in.x, [], batchSize));
if numel(layer.weights{2}) > 0
    out.x = bsxfun(@plus, out.x, layer.weights{2}(:));
end
out.x = reshape(out.x, layer.outHeight, layer.outWidth, layer.outChannels, batchSize);
end
