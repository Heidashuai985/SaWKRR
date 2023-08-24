function result=Sa_WKRR(input,xtrain,ytrain,C,l,option)
%% Description
% input: model input; xtrain: training input; ytrain: desired output;
% C: regularization coefficient l:kernel length
% option: develop KRR or KRVFL,specify 'KRR' for KRR, and 'KRVFL' for KRVFL
%% Calculation of regularization coefficient
imp=corrcoef([xtrain,ytrain]);
imp=abs(imp(1:end-1,end))';
imp=length(imp)*imp/sum(imp);
%% Main
mat1=(xtrain.*sqrt(imp))*(xtrain.*sqrt(imp))';
mat2=sum((xtrain.*sqrt(imp)).^2,2);
dis=mat2+mat2'-2*mat1;
K1=exp(-l*dis);
K2=(xtrain)*(xtrain)';
result=[];
for i=1:length(input(:,1))
    distance=sum(((xtrain-input(i,:)).^2).*imp,2);
    qt3=prctile(distance,75);
    qt1=prctile(distance,25);
    weights=ones(length(ytrain),1);
    idx=find(distance>=qt3);
    weights(idx)=length(ytrain)*distance(idx)/sum(distance);
    idx=find(distance<=qt1);
    weights(idx)=length(ytrain)*distance(idx)/sum(distance);
    idx=find(distance==0);
    weights(idx)=2;
    idx=find(weights>2);
    weights(idx)=2;   
    weights=weights.*eye(length(ytrain));
    inputk1=exp(-l*distance)';
    inputk2=(input(i,:))*(xtrain)';
    if strcmp(option,'KRR')
        output=inputk1*inv(eye(length(ytrain))/C+weights*K1)*weights*ytrain;
    elseif strcmp(option,'KRVFL')
        output=(inputk1+inputk2)*inv(eye(length(ytrain))/C+weights*(K1+K2))*weights*ytrain;
    end
    result=[result;output];
end
        