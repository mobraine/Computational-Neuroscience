% assign_classifer_gradDesc.m 
%
% demo of single-layer binary classifer, using gradient descent
%   sum-square error, sigmoid activation
%   batch mode
%   derived from lab exercise in Bruno Olshausen's VS298 / lab2s.m - single neuron learning
%   uses Bruno's data files, "apples" and "oranges"
% finds line, w0 + w1*x1 + w2*x2 = 0, to best separate 2 categories of data pts

clear all;   close all;     
fprintf(1,'\n\n\n\n');

addpath('data');   % "apples", "oranges"

rng('default');   % "standard" random number seed -> reproducible simulations

%load data files (from Bruno's VS298), and initialize data array
load apples;    load oranges;   % -> pair of 2x10 matrices

data=[apples oranges];          % -> 2x20 matrix
[N nDataPts]=size(data);               % N=2, K=20
 

% initialize teacher (1x20 array)
teacher = [0*ones(1,nDataPts/2) +1*ones(1,nDataPts/2)]; 
lambda = 1;    % sigmoid parameter, in activation function

% learning rate
eta = input('learning rate:   ');
if isempty(eta)  error('sorry you MUST specify a learning rate !');  end

nTrials = input('number of batch-mode trials (e.g. 500):   ');
if isempty(nTrials)  ;  nTrials=500;  end

alpha = input('alpha: ');
if isempty(alpha)  error('sorry you MUST specify alpha !');  end

% initialize weights
w  = randn(2,1);          % 2x1 array
w0 = randn(1);            % scalar

% initialize data plot, and display hyperplane implied by initial-guess weights
if isunix
    figHanMain = figure('position',[60 1000 300 600]);
elseif ispc
    figHanMain = figure('position',[60   60 300 600]);
else
    error('unrecognized operating system');
end

subplot(2,1,1);
plot(apples(1,:),apples(2,:),'b+',oranges(1,:),oranges(2,:),'ro');  hold on
x1=0:4;
x2=-(w(1)*x1+w0)/w(2);
axis([0 4 -1 3])
h=plot(x1,x2);   grid on;   drawnow;

errTrain = zeros(nTrials,1);     % initialize loss histories
wTrain = zeros(nTrials,2); % initialize weights histories
w0Train = zeros(nTrials,1); % same as above

for iTrial=1:nTrials    % loop over trials (iterations)
   % initialize dw's and loss_sum
   dw(1)  = 0;
   dw(2)  = 0;
   dw0    = 0;
   loss_sum = 0;
   
   % loop over training set
   for iDataPt=1:nDataPts
        % compute neuron output
        u = w0 + w(1)*data(1,iDataPt) + w(2)*data(2,iDataPt); 
        
        y = 1 / (1 + exp(-lambda*u));  % activation
            
        % compute error
        E = teacher(iDataPt) - y;     % scalar value of raw error
        
        % accumulate dw and dw0
        dw(1) = dw(1) + E*(exp(-lambda*u)/((1+exp(-lambda*u))^2))*data(1,iDataPt);
        dw(2) = dw(2) + E*(exp(-lambda*u)/((1+exp(-lambda*u))^2))*data(2,iDataPt);
        dw0   = dw0   + E*(exp(-lambda*u)/((1+exp(-lambda*u))^2));            
        
        % the term (exp(-lambda*u)/((1+exp(-lambda*u))^2)) is for what??
        % deleted
        % dw(1) = dw(1) + E*data(1,iDataPt);
        % dw(2) = dw(2) + E*data(2,iDataPt);
        % dw0   = dw0   + E;            

        loss = E^2;     % loss = error-squared
        
        % accumulate error
        loss_sum = loss_sum + loss;     % accumulated loss  
   end
   
   % update weights with weight decay
   w(1) = w(1) + eta*(dw(1) - alpha * w(1)); 
   w(2) = w(2) + eta*(dw(2) - alpha * w(2));     
   w0   = w0   + eta*(dw0 - alpha * w0);       
     
   errTrain(iTrial) = loss_sum; % record history of loss
   wTrain(iTrial, 1) = w(1);
   wTrain(iTrial, 2) = w(2);
   w0Train(iTrial) = w0;

   % update display of separating hyperplane
   x2=-(w(1)*x1+w0)/w(2);
   set(h,'YData',x2);       % "h" from above, "h=plot(...)"
         
   drawnow  
end
hold off
figure;
plot(errTrain);
xlabel('iteration');
ylabel('loss');
txt = {['Eta=' num2str(eta)], ['nTrials=' num2str(nTrials)], ['final loss=' num2str(loss_sum)]};
text(nTrials / 2, round(max((errTrain) + min(errTrain)) / 2),txt,Interpreter="latex");

figure;
subplot(3,1,1);
plot(wTrain(:,1));
xlabel('iteration');
ylabel('w1');
subplot(3,1,2);
plot(wTrain(:,2));
xlabel('iteration');
ylabel('w2');
txt = {['Eta=' num2str(eta)], ['nTrials=' num2str(nTrials)], ['final loss=' num2str(loss_sum)], ['alpha=' num2str(alpha)]};
text(nTrials / 2, round(max((wTrain(:,2)) + min(wTrain(:,2))) / 2),txt,Interpreter="latex");

subplot(3,1,3);
plot(w0Train);
xlabel('iteration');
ylabel('w0');

fprintf(1,'\nfinal loss = %f\n', loss_sum);





