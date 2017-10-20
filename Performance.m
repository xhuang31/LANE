function [F1macro,F1micro] = Performance(Xtrain,Xtest,Ytrain,Ytest)
%Evaluate the performance of classification for both multi-class and multi-label Classification
%     [F1macro,F1micro] = Performance(Xtrain,Xtest,Ytrain,Ytest)
%
%       Xtrain is the training data with row denotes instances, column denotes features
%       Xtest  is the test data with row denotes instances, column denotes features
%       Ytrain is the training labels with row denotes instances
%       Ytest  is the test labels
 
%   Copyright 2017, Xiao Huang and Jundong Li.
%   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

    %% Multi class Classification
    if size(Ytrain,2) == 1 && length(unique(Ytrain)) > 2
        t = templateSVM('Standardize',true);
        model = fitcecoc(Xtrain,Ytrain,'Learners',t);
        pred_label = predict(model,Xtest);
        [micro, macro] = micro_macro_PR(pred_label,Ytest);
        F1macro = macro.fscore;
        F1micro = micro.fscore;
       
    else
        %% For multi-label classification, computer micro and macro
        rng default % For repeatability
        NumLabel = size(Ytest,2);
        macroTP = zeros(NumLabel,1);
        macroFP = zeros(NumLabel,1);
        macroFN = zeros(NumLabel,1);
        macroF = zeros(NumLabel,1);
        for i = 1:NumLabel
            model = fitcsvm(Xtrain,Ytrain(:,i),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
            pred_label = predict(model,Xtest);
            mat = confusionmat(Ytest(:,i), pred_label);
            if size(mat,1) == 1
                macroTP(i) = sum(pred_label);
                macroFP(i) = 0;
                macroFN(i) = 0;
                if macroTP(i) ~= 0
                    macroF(i) = 1;
                end
            else
                macroTP(i) = mat(2,2);
                macroFP(i) = mat(1,2);
                macroFN(i) = mat(2,1);
                macroF(i) = 2*macroTP(i)/(2*macroTP(i)+macroFP(i)+macroFN(i));
            end  
        end
        F1macro = mean(macroF);
        F1micro = 2*sum(macroTP)/(2*sum(macroTP)+sum(macroFP)+sum(macroFN));
    end
end

function [micro, macro] = micro_macro_PR(pred_label,orig_label)
% computer micro and macro: precision, recall and fscore
    mat = confusionmat(orig_label, pred_label);
    len = size(mat,1);
    macroTP = zeros(len,1);
    macroFP = zeros(len,1);
    macroFN = zeros(len,1);
    macroP = zeros(len,1);
    macroR = zeros(len,1);
    macroF = zeros(len,1);
    for i = 1:len
        macroTP(i) = mat(i,i);
        macroFP(i) = sum(mat(:, i))-mat(i,i);
        macroFN(i) = sum(mat(i,:))-mat(i,i);
        macroP(i) = macroTP(i)/(macroTP(i)+macroFP(i));
        macroR(i) = macroTP(i)/(macroTP(i)+macroFN(i));
        macroF(i) = 2*macroP(i)*macroR(i)/(macroP(i)+macroR(i));
    end
%     macroP(isnan(macroP)) = 0;
%     macroR(isnan(macroR)) = 0;
    macroF(isnan(macroF)) = 0;
%     macro.precision = mean(macroP);
%     macro.recall = mean(macroR);
    macro.fscore = mean(macroF);

    micro.precision = sum(macroTP)/(sum(macroTP)+sum(macroFP));
    micro.recall = sum(macroTP)/(sum(macroTP)+sum(macroFN));
    micro.fscore = 2*micro.precision*micro.recall/(micro.precision+micro.recall);
end