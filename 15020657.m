
clc
C=1;
data = csvread('datafile.csv');

%% Changing 2,4 to -1,+1
data(data(:,11)==2,11) =-1;
data(data(:,11)==4,11) =1;


X = data(:,[2:10]); 
Y = data(:,[11]);
X = [ones(size(X, 1), 1) X];
X

%% Getting the first 2/3 of the data for training

X_Train = X([1:(rows(data)/3)*2],:);
Y_Train = Y([1:(rows(data)/3)*2],:);


%% Getting the last 1/3 of the data for testing

X_Test = X([((rows(data)/3)*2)+1:rows(data)],:);
Y_Test = Y([((rows(data)/3)*2)+1:rows(data)],:);


num_Training = rows(X_Train); %% Number of training set rows
num_Testing = rows(X_Test); %% Number of testing set rows

num_Iter = [1,2,10]; %% number of iterations
f = {"PA","PA-1","PA-2"}; %% different functions

for func_No = 1:columns(f)
  func = f{func_No}
  for j = 1:columns(num_Iter)

    %% number of iterations as defined
    iter = num_Iter(1,j)
    W = [0,0,0,0,0,0,0,0,0,0];

    for i=1:iter

      for row=1:num_Training
        row_X = X_Train(row,:);
        val_Y = Y_Train(row,:);
        
        %% Loss function
        l = max(0,1-val_Y*(W*row_X'));
        
        if (strcmp(func,"PA"))
          t = l/norm(row_X).^2;
        elseif (strcmp(func,"PA-1"))
          t = min(C,(l/(norm(row_X)).^2));
        elseif (strcmp(func,"PA-2"))
          t = l/(norm(row_X).^2)+(1/2*C);
        endif
        
        %% Calculating new weights
        W = W + t*val_Y*row_X;
        
      endfor
    endfor
    
    %%accuracy for the training dataset
    accuracy_Train = 0;
    for m = 1:num_Training
      train_Row_X = X_Train(m,:);
      trainVal_Y = Y_Train(m,:);
      y_Pred_Train = sign(W*train_Row_X');

      if(y_Pred_Train*trainVal_Y==1)
       accuracy_Train = accuracy_Train+1;
      endif
    endfor
    
    
    %% Accuracy for the testing dataset
    accuracy_Test = 0;
    for n = 1:num_Testing
      testRow_X = X_Test(n,:);
      testVal_Y = Y_Test(n,:);
      y_Pred_Test = sign(W*testRow_X');

      if(y_Pred_Test*testVal_Y==1)
       accuracy_Test = accuracy_Test+1;
      endif
     endfor
     printf("accuracy for training dataset: %f \n",(accuracy_Train/num_Training)*100)
     printf("accuracy for testing dataset: %f \n",(accuracy_Test/num_Testing)*100)

   endfor
   printf("--------------------------------------------\n")
endfor