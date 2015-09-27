function num = num_params(layers)
    % Compute the number of parameters in a neural network defined by a cell
    % array of layers.
    num = 0;
    for iLayer = 1:numel(layers)
        if  isfield(layers{iLayer}, 'weights')
            for iW = 1:numel(layers{iLayer}.weights)
                num = num + numel(layers{iLayer}.weights{iW});
            end
        end
        % Deprecated.
        if isfield(layers{iLayer}, 'filters')
            num = num + numel(layers{iLayer}.filters);
        end
        if isfield(layers{iLayer}, 'biases')
            num = num + numel(layers{iLayer}.biases);
        end
    end
end
