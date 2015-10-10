function vl_test_ttlayers(gpu, tests)
% VL_TEST_TTLAYERS Test the TT-layer with numeric differentiation
%    VL_TEST_TTLAYERS(0) Test the CPU implementation.
%    VL_TEST_TTLAYERS(1) Test the GPU implementation.

range = 100;

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) range * gpuArray.randn(varargin{:});
  grand = @(varargin) range * gpuArray.rand(varargin{:});
else
  grandn = @(varargin) range * randn(varargin{:});
  grand = @(varargin) range * rand(varargin{:});
end

switch gpu
  case 0,
    fprintf('testing the CPU code\n');
  case 1
    fprintf('testing the GPU code\n');
end

rng(1);

if nargin < 2
  tests = 1:3;
end

function y = vl_nntt_forward_weights(layer, in, out, iGroup, values)
  layer.weights{iGroup} = values;
  outIn = vl_nntt_forward(layer, in, out);
  y = outIn.x;
end

function y = vl_nntt_forward_x(layer, in, out, x)
  in.x = x;
  outIn = vl_nntt_forward(layer, in, out);
  y = outIn.x;
end

for l = tests
  fprintf('test number %d\n', l)
  % resets random number generator to obtain reproducible results
  if gpu
    parallel.gpu.rng(0, 'combRecursive');
  else
    rng(0, 'combRecursive');
  end
  switch l
    case 1
      disp('Testing vl_nntt_* with the identity TT-matrix.');

      in.x = grandn(8, 32, 3, 4, 'single');
      W = tt_ones([4, 4, 4, 4, 3]);
      W.core = single(W.core);
      W = diag(W);
      layer.W = W;
      layer.weights{1} = W.core;
      if gpu
        layer.weights{1} = gpuArray(layer.weights{1});
      end
      layer.outHeight = 8;
      layer.outWidth = 32;
      layer.outChannels = 3;
      layer.weights{2} = grandn(8 * 32 * 3, 1, 'single');
      out = [];
      out = vl_nntt_forward(layer, in, out);
      y = out.x;
      out.dzdx = grandn(size(y), 'single');
      in = vl_nntt_backward(layer, in, out);
      for iGroup = 1:numel(layer.weights)
          vl_testder(@(w) vl_nntt_forward_weights(layer, in, out, iGroup, w), layer.weights{iGroup}, out.dzdx, in.dzdw{iGroup}, range * 1e-2);
      end
      vl_testder(@(x) vl_nntt_forward_x(layer, in, out, x), in.x, out.dzdx, in.dzdx, range * 1e-2);

    case 2
      disp('Testing vl_nntt_* with a random square TT-matrix.');
      % Shape for the input and output tensors.
      tensorShape = [4, 4, 4, 4, 3];
      batchSize = 10;
      ranks = [1, 4, 6, 10, 5, 1];
      W = tt_rand(tensorShape.^2, 5, ranks);
      W.core = single(W.core);
      W = tt_matrix(W, tensorShape, tensorShape);
      layer.W = W;
      layer.weights{1} = W.core;
      if gpu
        layer.weights{1} = gpuArray(layer.weights{1});
      end
      layer.outHeight = 8;
      layer.outWidth = 32;
      layer.outChannels = 3;
      in.x = grandn(8, 32, 3, batchSize, 'single');
      layer.weights{2} = grandn(8 * 32 * 3, 1, 'single');
      out = [];
      out = vl_nntt_forward(layer, in, out);
      y = out.x;
      exactY = full(W) * reshape(in.x, [], batchSize);
      exactY = bsxfun(@plus, exactY, layer.weights{2});
      vl_testsim(y, reshape(exactY, 8, 32, 3, batchSize));
      out.dzdx = grandn(size(y), 'single');
      in = vl_nntt_backward(layer, in, out);
      for iGroup = 1:numel(layer.weights)
          vl_testder(@(w) vl_nntt_forward_weights(layer, in, out, iGroup, w), layer.weights{iGroup}, out.dzdx, in.dzdw{iGroup}, range * 1e-2);
      end
      vl_testder(@(x) vl_nntt_forward_x(layer, in, out, x), in.x, out.dzdx, in.dzdx, range * 1e-2);

    case 3
      disp('Testing vl_nntt_* with random rectangular TT-matrices.');
      for bias = [false true]
        for batchSize = [1 3]
          inputTensorShape = [3, 6, 4, 5];
          outputTensorShape = [4, 11, 7, 13];
          layer.outHeight = 2 * 11;
          layer.outWidth = 7 * 13;
          layer.outChannels = 2;
          ranks = [1, 5, 9, 5, 1];
          W = tt_rand(outputTensorShape .* inputTensorShape, length(inputTensorShape), ranks, []);
          W.core = single(W.core);
          W = tt_matrix(W, outputTensorShape, inputTensorShape);
          layer.W = W;
          layer.weights{1} = W.core;
          if gpu
            layer.weights{1} = gpuArray(layer.weights{1});
          end
          if bias
            layer.weights{2} = grandn(prod(outputTensorShape), 1, 'single');
          else
            layer.weights{2} = [];
          end
          in.x = grandn(9, 8, 5, batchSize, 'single');
          out = [];
          layer.W
          size(in.x)
          out = vl_nntt_forward(layer, in, out);
          y = out.x;
          exactY = full(W) * reshape(in.x, [], batchSize);
          if bias
            exactY = bsxfun(@plus, exactY, layer.weights{2});
          end
          vl_testsim(y, reshape(exactY, layer.outHeight, layer.outWidth, layer.outChannels, batchSize));
          out.dzdx = grandn(size(y), 'single');
          in = vl_nntt_backward(layer, in, out);
          for iGroup = 1:numel(layer.weights)
              vl_testder(@(w) vl_nntt_forward_weights(layer, in, out, iGroup, w), layer.weights{iGroup}, out.dzdx, in.dzdw{iGroup}, range * 1e-2);
          end
          vl_testder(@(x) vl_nntt_forward_x(layer, in, out, x), in.x, out.dzdx, in.dzdx, range * 1e-2);
        end
      end
    end
  end
end
end
