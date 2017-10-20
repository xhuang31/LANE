% Code for Label Informed Attributed Network Embedding
% 
%   Copyright 2017, Xiao Huang and Jundong Li.
%   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

Dataset = 'BlogCatalog';
if strcmp(Dataset,'BlogCatalog')
    load('BlogCatalog.mat')
    alpha1 = 43; % the weight for node attribute information
    alpha2 = 36; % the weight for label information
    numiter = 5; % the max number of iteration
    delta1 = 0.97; % weight of network information for constructing test representation H2
    delta2 = 1.6; % weight of node attribute information for constructing test representation H2
elseif strcmp(Dataset,'Flickr')
    load('Flickr.mat')
    alpha1 = 10^0.8; % the weight for node attribute information
    alpha2 = 100; % the weight for label information
    numiter = 4; % the max number of iteration
    delta1 = 0.3; % weight of network information for constructing test representation H2
    delta2 = 2.3; % weight of node attribute information for constructing test representation H2
end

d = 100; % the dimension of the embedding representation
G = Network;
A = Attributes;
clear Attributes & Network
[n,~] = size(G); % Total number of nodes
G(1:n+1:n^2) = 1;
Y = [];
LabelIdx = unique(Label); % Indexes of all label categories
for n_Label_i = 1:length(LabelIdx)
    Y = [Y,Label==LabelIdx(n_Label_i)];
end
Y=Y*1;

Indices = crossvalind('Kfold',n,20); % 5-fold cross-validation indices
Group1 = find(Indices <= 8); % 1 for 1/16, 2 for 1/8, 4 for 1/4, 16 for 100% of training group
Group2 = find(Indices >= 17); % test group
%% Training group
G1 = sparse(G(Group1,Group1)); % network of nodes in the training group
A1 = sparse(A(Group1,:)); % node attributes of nodes in the training group
Y1 = sparse(Y(Group1,:)); % labels of nodes in the training group
%% Test group
A2 = sparse(A(Group2,:)); % node attributes of nodes in the test group
GC1 = sparse(G(Group1,:)); % For constructing test representation H2
GC2 = sparse(G(Group2,:)); % For constructing test representation H2

%% Label Informed Attributed Network Embedding (Supervised)
disp('Label informed Attributed Network Embedding (LANE), 5-fold with 50% of training is used:')
H1 = LANE_fun(G1,A1,Y1,d,alpha1,alpha2,numiter); % representation of training group
H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1)); % representation of test group
[F1macro1,F1micro1] = Performance(H1,H2,Label(Group1,:),Label(Group2,:)) %


%% Unsupervised Attributed Network Embedding (LANE w/o Label)
disp('Unsupervised Attributed Network Embedding (LANE w/o Label):')
if strcmp(Dataset,'BlogCatalog')
    % Parameters of BlogCatalog in Unsupervised
    beta1 = 8; % the weight for node attribute information
    beta2 = 0.1; % the weight for the correlations
    numiter = 3; % the max number of iteration
    delta1 = 1.4; % weight of network information for constructing test representation H2
    delta2 = 1; % weight of node attribute information for constructing test representation H2
elseif strcmp(Dataset,'Flickr')
    % Parameters of BlogCatalog in Unsupervised
    beta1 = 0.51; % the weight for node attribute information
    beta2 = 0.1; % the weight for the correlations
    numiter = 2; % the max number of iteration
    delta1 = 0.55; % weight of network information for constructing test representation H2
    delta2 = 2.1; % weight of node attribute information for constructing test representation H2
end

H1 = LANE_fun(G1,A1,d,beta1,beta2,numiter); % representation of training group
H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1)); % representation of test group
[F1macro2,F1micro2] = Performance(H1,H2,Label(Group1,:),Label(Group2,:)) %