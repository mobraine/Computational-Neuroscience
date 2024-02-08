% assign_2d_RF_sysIdent.m
%
% simulated estimation of 2d spatial receptive field of visual cortex simple cell
% uses 'scg' (scaled conjugate gradient) optimization, so no need to specify step size
% simple regression with netlab, using a glm, to estimate Gabor-RF (simulated neuron/filter response waveform)
% glm-fit uses L2 (ridge) regression, with regularization parameter "alpha"
% use training dataset to get estimate of RF map 
%   using this estimated model, evaluates VAF on holdback validationdataset, for a given alpha

%   figure 1:  example stimulus images
%   figure 2:  model filter, and estimate of rfMap
 

% needs to run:
%   netlab - folder of functions from netlab toolbox
%   getStimulusMovies.m - construct movies, or read from files
%   makeModelRF.m - create linear 2d gabor filter, for model of receptive field
%   imagescZadj.m - imagesc, but force mid-range of color-map to be at zero
%   hwr.m - half wave rectification

clear all;  fprintf(1,'\n\n\n\n\n\n');   close all;  

rng('default');  % "standard" random number seed -> reproducible simulations 

addpath('netlab3_3');

%  choose stimulus   %%%%%%%%%%%%%%%
commandwindow
stimType = input('stimulus:  white noise (1) or natural images (2):  ');
if stimType==1
    option.stimulus = 'white';
else
    option.stimulus = 'McGill_clips';
end
clear stimType;

algoType = input('stimulus:  cross-correlation (1) or scg-regression (2):  ');
if algoType==1
    option.algorithm = 'crossCorr';
else
    option.algorithm = 'scg';
end
clear algoType;

% partion data:  most for training, some for validation 
nMovies.train  = 4;
nMovies.valid = 1; 
nMovies.test = 1;
nMovies.total = nMovies.train + nMovies.valid + nMovies.test; 
if (nMovies.total > 6)
    error('sorry, we only have 6 movie response datasets');
end

imgSiz  = 32;     % width/height of stimulus/filter/rfMap  
nPixels = imgSiz^2; 

durSec = 5;   refreshHz = 75;  % simulate 5 seconds at 75 hz frame rate
nFrames = durSec*refreshHz;  % e.g. 5 sec at 75 hz  (="ndata" in netlab)
if nFrames>375
    error('too many frames for these movie files !');
end

% specify model receptive field (Gabor function followed by half-power law)
model.lambda = 8;
model.phase  = 0;
model.ori    = 0;  
model.pwrExp = 2; % input('power law exponent:   ');

%  graph specs
fig.stim.pos          = [400 200 300 300];     %[xOff yOff xSize ySize];
fig.stim.handle       = figure('position',fig.stim.pos,'toolbar','none','menubar','none','name','stimulus');
fig.model.pos         = [50 600 300 400];  
fig.model.handle      = figure('position',fig.model.pos,'toolbar','none','menubar','none','name','model');

%  create model filter, and plot in Figure 1 
rfModel = makeModelRF(model,imgSiz);    % creates model filter (Gabor function)
rfModelVec  = reshape(rfModel,1,nPixels);      % make a 1d version, for later use            

%  partition full dataset into 3 subsets, for training and validation and
%  tesitng
stimMovie      = zeros(nPixels,nFrames);
stimMovieTrain   = [];
stimMovieValid  = [];
stimMovieTest  = [];
respTrain      = [];
respValid     = [];
respTest     = [];
for iMovie=1:nMovies.total
    getStimulusMovies;            % -> stimMovie = nPixels x nFrames, range -1 to +1
    output = rfModelVec*stimMovie;    % linear filter response to the stimulus    
    output = hwr(output);  % half-wave rectify (set negative values to zero)
    output = output.^model.pwrExp;  % power-low for positive values
                   
    % accumulate results in dataset partitions: 
    if iMovie<=nMovies.train
        stimMovieTrain = [stimMovieTrain stimMovie]; % nPixels x nFrames*nMovies.train
        respTrain    = [respTrain    output];        %     1   x nFrames*nMovies.train  
    elseif (nMovies.train + nMovies.valid <= iMovie) && (iMovie < nMovies.total)
        stimMovieValid  = [stimMovieValid stimMovie];  % nPixels x nFrames*nMovies.valid
        respValid     = [respValid    output];          %     1   x nFrames*nMovies.valid 
    else
        stimMovieTest  = [stimMovieTest stimMovie];  % nPixels x nFrames*nMovies.valid
        respTest     = [respTest    output];          %     1   x nFrames*nMovies.valid
    end                      
end  % end of iMovie-loop


if strcmp(option.algorithm,'scg')
    % initialize options for optimization
    nin  = imgSiz^2; % number of inputs
    nout = 1;        % number of outputs:  one neuron
    netOptions     = zeros (1,18); 
    netOptions(1)  = 0;
    netOptions(2)  = .0001;      % termination criterion: distance moved
    netOptions(3)  = netOptions(2); % for scg, use VERY small value, eg 10^-9
    netOptions(14) = 200;    % max no of iterations - should be >= no of dim.s ?
    
    % commandwindow
    % alpha = input('alpha:  ');   % regularization (alpha-loop should begin here)
    alphas = [0.1, 1, 10, 100, 1000, 10000];
    vafTrain = [];
    vafValid = [];
    vafTest = [];
    for alpha = alphas
        % estimate rfMap, for this alpha
        net = glm(nin, nout,'linear',alpha);       % initialize structure  
        net.w1 = 0*net.w1;    net.b1 = 0*net.b1;   % sparse prior
        [net, netOptions] = netopt(net,netOptions,stimMovieTrain',respTrain','scg');
        rfMap2d  = reshape(net.w1,imgSiz,imgSiz);  % reshape to 2d 
        rfMapVec    = reshape(rfMap2d,nPixels,1);    % make a 1-d version
        
        predRespTrain = rfMapVec'*stimMovieTrain; 
        vaf.R_matrix = corrcoef(respTrain,predRespTrain);  % -> 2x2 matrix, ones on diagonal
        vaf.offDiag = vaf.R_matrix(1,2);
        vaf.vaf = vaf.offDiag^2.;
        vafTrain = [vafTrain, 100*vaf.vaf]; 
        
        predRespValid = rfMapVec'*stimMovieValid; 
        vaf.R_matrix = corrcoef(respValid,predRespValid);  % -> 2x2 matrix, ones on diagonal
        vaf.offDiag = vaf.R_matrix(1,2);
        vaf.vaf = vaf.offDiag^2.;
        vafValid = [vafValid, 100*vaf.vaf]; 
        
        predRespTest = rfMapVec'*stimMovieTest; 
        vaf.R_matrix = corrcoef(respTest,predRespTest);  % -> 2x2 matrix, ones on diagonal
        vaf.offDiag = vaf.R_matrix(1,2);
        vaf.vaf = vaf.offDiag^2.;
        vafTest = [vafTest, 100*vaf.vaf]; 
        
    end
    figure;
    hold on;
    semilogx(logspace(-1,4,6),vafTrain,'r');
    semilogx(logspace(-1,4,6),vafValid,'b');
    semilogx(logspace(-1,4,6),vafTest,'g');
    title(option.stimulus);
    xlabel('alpha');
    ylabel('VAF');
    legend('traing', 'validation', 'test');
    % show some example stimulus images
    figure(fig.stim.handle);
    for ix=1:4
        stimImgVec = stimMovie(:,ix+20);
        stimImg = reshape(stimImgVec,imgSiz,imgSiz);
        subplot(2,2,ix);
        imagescZadj(stimImg); axis image; axis off; colormap('gray');
    end
    
    % Task4B 
    % graph estimated rfMap, below "actual" (model) receptive field:
    figure(fig.model.handle); 
    % subplot(4,1,1);
    % imagescZadj(rfModel);  hold on; axis image; axis off; colorbar; title('RF model filter');
    
    globalMin = Inf; % Initialize global minimum
    globalMax = -Inf; % Initialize global maximum
    
    alphas = [0.1, 100, 10000]; % Define alpha values
    counter = 1;
    
    for alpha = alphas
        % estimate rfMap, for this alpha
        net = glm(nin, nout,'linear',alpha);       % initialize structure  
        net.w1 = 0*net.w1;    net.b1 = 0*net.b1;   % sparse prior
        [net, netOptions] = netopt(net,netOptions,stimMovieTrain',respTrain','scg');
        rfMap2d  = reshape(net.w1,imgSiz,imgSiz);
        
        % Update global min and max
        globalMin = min(globalMin, min(rfMap2d(:)));
        globalMax = max(globalMax, max(rfMap2d(:)));
    end
    subplot(4,1,1);
    imagesc(rfModel, [globalMin, globalMax]);  hold on; axis image; axis off; colorbar; title('RF model filter');
    for alpha = alphas
        % estimate rfMap, for this alpha
        net = glm(nin, nout,'linear',alpha);       % initialize structure  
        net.w1 = 0*net.w1;    net.b1 = 0*net.b1;   % sparse prior
        [net, netOptions] = netopt(net,netOptions,stimMovieTrain',respTrain','scg');
        rfMap2d  = reshape(net.w1,imgSiz,imgSiz);  % reshape to 2d
        
        subplot(4,1,1+counter);  
        imagesc(rfMap2d,[globalMin, globalMax]); 
        if strcmp(option.algorithm,'scg')
            title(sprintf('scg RF estimate for alpha = %3.1f',alpha));  
        else
            title(sprintf('RF estimate for cross-correlation'));      
        end
        axis image;  axis off;  colorbar; 
        counter = counter+1;
    end

elseif strcmp(option.algorithm,'crossCorr')
    nLags=1; maxLag=0;  % (settings to make xcorr.m give us what we want)
    crossCorrAll = zeros(nPixels,nMovies.train*nFrames);
    for iPix=1:nPixels
        crossCorrAll(iPix,:) = xcorr(respTrain,stimMovieTrain(iPix,:),maxLag,'unbiased');  % cross-correlation
    end
    rfMap = crossCorrAll(:,end-nLags+1:end);  % only take positive lagged values
    rfMap2d = reshape(rfMap,imgSiz,imgSiz);  % reshape into 2d
    rfMapVec  = reshape(rfMap2d,nPixels,1);    % make a 1-d version
    clear iPix crossCorrAll rfMap nLags maxLag;

    % show some example stimulus images
    figure(fig.stim.handle);
    for ix=1:4
        stimImgVec = stimMovie(:,ix+20);
        stimImg = reshape(stimImgVec,imgSiz,imgSiz);
        subplot(2,2,ix);
        imagescZadj(stimImg); axis image; axis off; colormap('gray');
    end
    
    % graph estimated rfMap, below "actual" (model) receptive field:
    figure(fig.model.handle); 
    subplot(2,1,1);
    imagescZadj(rfModel);  hold on; axis image; axis off; colorbar; title('RF model filter');
    subplot(2,1,2);  
    imagescZadj(rfMap2d); 
    if strcmp(option.algorithm,'scg')
        title(sprintf('scg RF estimate for alpha = %3.1f',alpha));  
    else
        title(sprintf('RF estimate for cross-correlation'));      
    end
    axis image;  axis off;  colorbar;  

else
    error('unrecognized algorithm');
end



% use rfMap estimate to generate prediction of the training and validation responses)
predRespTrain = rfMapVec'*stimMovieTrain; 
predRespValid = rfMapVec'*stimMovieValid; 
predRespTest = rfMapVec'*stimMovieTest; 

residValidNew = respValid - predRespValid;  % residual - error in prediction of validation response

% calculate VAF for validation dataset
vaf.R_matrix = corrcoef(respValid,predRespValid);  % -> 2x2 matrix, ones on diagonal
vaf.offDiag = vaf.R_matrix(1,2);
vaf.vaf = vaf.offDiag^2.;
fprintf(1,'\nVAF for validation dataset = %5.1f percent\n', 100*vaf.vaf);   

% calculate VAF for test dataset
vaf.R_matrix = corrcoef(respTest,predRespTest);  % -> 2x2 matrix, ones on diagonal
vaf.offDiag = vaf.R_matrix(1,2);
vaf.vaf = vaf.offDiag^2.;
fprintf(1,'\nVAF for testing dataset = %5.1f percent\n', 100*vaf.vaf);   

