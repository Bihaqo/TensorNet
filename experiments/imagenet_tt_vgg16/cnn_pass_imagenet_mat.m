function [net, info] = cnn_pass_imagenet_mat(net, imdb, getBatch, outputDir, varargin)
%CNN_PASS_IMAGENET_MAT pass data through the network

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.figuresPath = fullfile('data','figures','fig') ;
opts.conserveMemory = true ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.useGpu = false;
opts.momentum = 0.9 ;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
%                                                         	   Validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

startTime = tic;
info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;
info.time = [] ;
opts.expDir
res = [] ;
%---------------data saving data 
dataDirMimic = outputDir;
if ~exist(dataDirMimic, 'dir')
    mkdir(dataDirMimic);
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
end

for epoch=1:opts.numEpochs
  val = opts.val ;

  info.train.objective(end+1) = 0 ;
  info.train.error(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.train.speed(end+1) = 0 ;
  info.val.objective(end+1) = 0 ;
  info.val.error(end+1) = 0 ;
  info.val.topFiveError(end+1) = 0 ;
  info.val.speed(end+1) = 0 ;
  info.time(end+1) = 0;

  % evaluation on validation set
  lastProcessBatch = 1;
  curBatchnumber = 0;  
  for t=lastProcessBatch:opts.batchSize:numel(val)
    curBatchnumber = curBatchnumber + 1;
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    if opts.prefetch
      nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      im = gpuArray(im) ;
    end
    
    net.layers{end}.class = labels ;
    opts.useGpu
    [res, data_img] = vl_simplenn_imagenet_mat(net, im, [], res, ...
      'disableDropout', true, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync) ; %#ok
    lastProcessBatch = t; %#ok
    save(fullfile(outputDir,'curBatch.mat'),'lastProcessBatch')
    save(fullfile(outputDir, strcat('data_img', num2str(curBatchnumber))), 'data_img');  
    % print information
    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time;
    info.val = updateError(opts, info.val, net, res, batch_time) ;
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    n = t + numel(batch) - 1 ;
    if strcmp(opts.errorType, 'mse')
        fprintf(' err %.5f\n', info.val.error(end)/n) ;
    else
        fprintf(' err %.1f err5 %.1f', ...
          info.val.error(end)/n*100, info.val.topFiveError(end)/n*100) ;
        fprintf('\n') ;
    end
  end

end 

% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res, speed)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
info.speed(end) = info.speed(end) + speed ;
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
  case 'mse'
      error = predictions - labels;
      info.error(end) = info.error(end) + sum(error(:).^2);
      
end

% -------------------------------------------------------------------------

