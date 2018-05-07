clear;
%positive example = 1
%negative examples = 0
data= dlmread('diabetes.csv',',',1,1);
X_test = data(500:end,1:end-1);
Y_test = data(500:end,end);
data=data(1:499,:);
%counting number of instances for each positive and negative call to
%identify minority class
posex=sum(data(:,8));
negex=size(data,1)-posex;
Nmaj=0;
Nmin=0;
minority_label=0;
if(posex>negex)
    minority_label=1;
    Nmaj=posex;
    Nmin=negex;
else
    minority_label=0;
    Nmaj=negex;
    Nmin=posex;
end
minorityCase = data(data(:,8) == minority_label,:);
minorityCase = minorityCase(:,1:end-1);
%%generation of synthetic data and their label using SMOTE
synthetic=SMOTE(minorityCase,1,25);
P=size(synthetic,1);
syn_label(1:P,1)=minority_label;
original_label=data(:,8);
original = data(:,1:end-1);
N=size(original,1);
synfrom=size(data,1)+1;
X_train = [original ; synthetic];
Y_train = [original_label;syn_label];
clear data data_label syn_label posex negex;

%%kernal function to calculate Gram matrix or kernal matrix

kernal_fun = @(X,Y) X*Y';
%% kernal matrix for original instances
K1 = kernal_fun(original,original);

%% K2 are inner product between original instance and synthetic instance

K2 = kernal_fun(original,synthetic);
%% K3 are dot product of synthetic sample

K3 = kernal_fun(synthetic,synthetic);
%% the kernel matrix K obtained by the addition on P new examples of minority class
K = [K1 K2;K2' K3];
%% adding sample serial number
K = [(1:size(K,1))' K];
clear K1 K2 K3;
%% the weighting factor Cmaj, Cmin, Csyn control the cost of missclassifying maj, min and syn instances
Cmin=Nmaj/N;     %cmin=Nmaj/N
Cmaj=Nmin/N;     %cmaj= Nmin/N
Csyn=Nmaj/N;     %csyn = Nmaj/N
C=[];
for i = 1:N
    if(Y_train(i,1)== minority_label)
        C = [C; Cmin];
    else
        C = [C;Cmaj];
    end
end
for i = N+1: size(X_train,1)
    C =[C;Csyn];
end
model = svmtrain(C,Y_train,K,'-t 4');
[predicted_label, accuracy,prob_estimates] = svmpredict(Y_test, X_test, model);
result = confusionmat(Y_test,predicted_label)



