% Sort the results into vectors
type1 = [];
type2 = [];

% Apply each vector in the training set to obtain a plot of classifications
for j = 1:number_to_train
    traincase = j;
    trainslice = input_data(traincase,:)';
    result = neural_net.forward(trainslice);
    known = output_data(traincase);
    switch known
        case 0
            type1 = [type1, result];
        case 1
            type2 = [type2, result];
    end
end

% Plot all training set classifications
close all
hold on
h1 = histogram(type1);
h2 = histogram(type2, h1.BinEdges);

xlabel('classification')
ylabel('number of observations')
legend('Normal', 'Suspect or Pathologic')


%Now we will feed the data from the test set into the neural network, to
%classify the observations as healthy or pathologic. In order to get a
%meaningful probability of an abnormal state, we must assume that abnormal
%states are equally prevalent in both the test and training data.
TN = 0;
TP = 0;
FN = 0;
FP = 0;
for j = number_to_train+1:2126
    
    testcase = j;
    testslice = input_data(testcase,:)';
    result = neural_net.forward(testslice);
    known = output_data(testcase);
    
    bin1 = find(h1.BinEdges >= result,1);
    bin2 = find(h2.BinEdges >= result,1);
    bin1 = min(bin1, h1.NumBins);
    bin2 = min(bin2, h2.NumBins);
    
    countNeg = h1.Values(bin1);
    countPos = h2.Values(bin2);
    probPositive = countPos/(countNeg+countPos);
    
    predictedValue = 0;
    if(isnan(probPositive) || probPositive >= 0.5)
        predictedValue = 1;
    end
    
    if known == predictedValue
        if known == 1
            TP = TP + 1;
        else
            TN = TN + 1;
        end
    else
        if known == 1
            FN = FN + 1;
        else
            FP = FP + 1;
        end
    end
    
end

%We will measure our success by the percent of test cases identified
%correctly and the percent of positive test cases identified correctly.
total_accuracy = (TP + TN)/(TP + TN + FP + FN)
PPV = TP/(TP + FP)
FDR = 1-PPV
NPV = TN/(TN + FN)
FOR = 1-NPV