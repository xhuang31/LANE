function H = LANE_fun(Net,Attri,LabelY,d,alpha1,alpha2,varargin)
%Jointly embed labels and attriuted network into embedding representation H
%     H = LANE_fun(Net,Attri,LabelY,d,alpha1,alpha2,numiter);
%     H = AANE_fun(Net,Attri,d,alpha1,alpha2,numiter);
% 
%          Net   is the weighted adjacency matrix
%         Attri  is the attribute information matrix with row denotes nodes
%        LabelY  is the label information matrix
%          d     is the dimension of the embedding representation
%         alpha1 is the weight for node attribute information
%         alpha2 is the weight for label information
%        numiter is the max number of iteration

%   Copyright 2017, Xiao Huang and Jundong Li.
%   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

n = size(Net,1);
LG = norLap(Net); % Normalized Network Laplacian
LA = norLap(Attri); % Normalized Node Attributes Laplacian
UAUAT = zeros(n,n); % UA*UA^T

if isempty(varargin)
    %% Unsupervised attriuted network embedding
    % Input of Parameters
    numiter = alpha2; % the max number of iteration
    beta1 = d; % the weight for node attribute information
    beta2 = alpha1; % the weight for the correlations
    d = LabelY; % the dimension of the embedding representation
    H = zeros(n,d);
    for i = 1:numiter
        HHT = H*H';
        TotalLG1 = LG+beta2*UAUAT+HHT;
        [UG,~] = eigs(.5*(TotalLG1+TotalLG1'),d);
        UGUGT = UG*UG';
        
        TotalLA = beta1*LA+beta2*UGUGT+HHT;
        [UA,~] = eigs(.5*(TotalLA+TotalLA'),d);
        UAUAT = UA*UA';
        
        TotalLH = UAUAT+UGUGT;
        [H,~] = eigs(.5*(TotalLH+TotalLH'),d);
    end
else
    %% Supervised attriuted network embedding
    numiter = varargin{1}; % the max number of iteration
    H = zeros(n,d);
    LY = norLap(LabelY*LabelY'); % Normalized Label Laplacian
    UYUYT = zeros(n,n); % UY*UY^T
    % Iterations
    for i = 1:numiter
        HHT = H*H';
        TotalLG1 = LG+alpha1*UAUAT+alpha2*UYUYT+HHT;
        [UG,~] = eigs(.5*(TotalLG1+TotalLG1'),d);
        UGUGT = UG*UG';
        
        TotalLA = alpha1*(LA+UGUGT)+HHT;
        [UA,~] = eigs(.5*(TotalLA+TotalLA'),d);
        UAUAT = UA*UA';
        
        TotalLY = alpha2*(LY+UGUGT)+HHT;
        [UY,~] = eigs(.5*(TotalLY+TotalLY'),d);
        UYUYT = UY*UY';
        
        TotalLH = UAUAT+UGUGT+UYUYT;
        [H,~] = eigs(.5*(TotalLH+TotalLH'),d);
    end
end
end

    function LapX = norLap(InpX)
    % Compute normalized graph Laplacian of InpX
        InpX = InpX'; % Transpose for speedup
        InpX = bsxfun(@rdivide,InpX,sum(InpX.^2).^.5); % Normalize
        InpX(isnan(InpX)) = 0;
        SX = InpX'*InpX;
        nX = length(SX);
        SX(1:nX+1:nX^2) = 1+10^-6;
        DXInv = spdiags(full(sum(SX,2)).^(-.5),0,nX,nX);
        LapX = DXInv*SX*DXInv;
        LapX = .5*(LapX+LapX');
    end

