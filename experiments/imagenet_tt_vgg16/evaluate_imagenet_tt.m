function info = evaluate_imagenet_tt(varargin)
% evaluate_imagenet_tt   evauate vgg16-tt models on imagenet
opts.data_path =  fullfile('data', 'imagenet12','data_path') ;
opts.dataDir = fullfile('data', 'imagenet12') ;
opts.expDir = fullfile('data', 'imagenet12-eval-vgg-f') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile('data', 'models', 'imagenet-vgg-f.mat') ;
opts.lite = false ;
opts.numFetchThreads = 8 ;
opts.train.batchSize = 64 ;
opts.train.numEpochs = 1 ;
opts.train.gpus = [1];
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss

% Synchronize label indexes between the model and the image database
imdb = cnn_imagenet_sync_labels(imdb, net);

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

data_path = fullfile(opts.data_path, 'data_img');
getBatchWrapper = @(imdb,batch) getBatch(imdb, batch, data_path);
[net,info] = cnn_train(net, imdb, getBatchWrapper, opts.train, ...
  'conserveMemory', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==2)) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, data_path)
% -------------------------------------------------------------------------
%data_path = '../../data_permanent/data_imagenet_mimic_deep/data_img';
sizeBatch = 64;
%batch(1)
for i = 1 : numel(batch)
%for val without train
   nFile =  ceil((batch(i) - 1281167) / 64);
   %nEl = batch(i) - 64 * floor((batch(i) - 1281167) / 64);

%    nFile =  ceil(batch(i) / sizeBatch);
%  fprintf('nFile %d\n', nFile);
   batchData = load(strcat(data_path, num2str(nFile), '.mat')); 
%batchData
   batchData = batchData.data_img;
   if i == 1
      im_size = size(batchData);
      im_size(4) = numel(batch);
      im =single(zeros(im_size)); 
   end
%	batch(i) - (nFile + 20019-1 - 1) * sizeBatch
%    batch(i)
%size(batchData)
%for val without train
   im(:,:,:,i) = batchData(:,:,:, batch(i) - 15 - (nFile + 20019-1 - 1) * sizeBatch);	
end
labels = imdb.images.label(batch) ;
%size(labels)
%size(im)
