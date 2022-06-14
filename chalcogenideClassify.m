%% Classification
clear,clc,close all;

load 'chalcogenidedGVal.mat';

reshapeSize=size(dG1M.interictal(1).deltaG1,1)*size(dG1M.interictal(1).deltaG1,2);

%Reshaping the dG maps for plotting in the dG plane and dG space
dG1_ii_1=reshape(dG1M.interictal(1).deltaG1,[reshapeSize,1]);
dG1_ii_10=reshape(dG1M.interictal(2).deltaG1,[reshapeSize,1]);
dG1_ii_20=reshape(dG1M.interictal(3).deltaG1,[reshapeSize,1]);
dG1_ii_73=reshape(dG1M.interictal(4).deltaG1,[reshapeSize,1]);
dG1_ii_159=reshape(dG1M.interictal(5).deltaG1,[reshapeSize,1]);
dG1_ii_239=reshape(dG1M.interictal(6).deltaG1,[reshapeSize,1]);
dG1_ii_358=reshape(dG1M.interictal(7).deltaG1,[reshapeSize,1]);
dG1_ii_479=reshape(dG1M.interictal(8).deltaG1,[reshapeSize,1]);
dG1_ii_498=reshape(dG1M.interictal(9).deltaG1,[reshapeSize,1]);
dG1_ii_500=reshape(dG1M.interictal(10).deltaG1,[reshapeSize,1]);

dG1_pi_1=reshape(dG1M.preictal(1).deltaG1,[reshapeSize,1]);
dG1_pi_5=reshape(dG1M.preictal(2).deltaG1,[reshapeSize,1]);
dG1_pi_10=reshape(dG1M.preictal(3).deltaG1,[reshapeSize,1]);
dG1_pi_15=reshape(dG1M.preictal(4).deltaG1,[reshapeSize,1]);
dG1_pi_20=reshape(dG1M.preictal(5).deltaG1,[reshapeSize,1]);
dG1_pi_25=reshape(dG1M.preictal(6).deltaG1,[reshapeSize,1]);
dG1_pi_30=reshape(dG1M.preictal(7).deltaG1,[reshapeSize,1]);
dG1_pi_35=reshape(dG1M.preictal(8).deltaG1,[reshapeSize,1]);
dG1_pi_40=reshape(dG1M.preictal(9).deltaG1,[reshapeSize,1]);
dG1_pi_42=reshape(dG1M.preictal(10).deltaG1,[reshapeSize,1]);

dG1ii_10k=[dG1_ii_1;dG1_ii_10;dG1_ii_20;dG1_ii_73;dG1_ii_159;dG1_ii_239;dG1_ii_358;dG1_ii_479;dG1_ii_498;dG1_ii_500];
dG1pi_10k=[dG1_pi_1;dG1_pi_5;dG1_pi_10;dG1_pi_15;dG1_pi_20;dG1_pi_25;dG1_pi_30;dG1_pi_35;dG1_pi_40;dG1_pi_42];

dG2_ii_1=reshape(dG1M.interictal(1).deltaG2,[reshapeSize,1]);
dG2_ii_10=reshape(dG1M.interictal(2).deltaG2,[reshapeSize,1]);
dG2_ii_20=reshape(dG1M.interictal(3).deltaG2,[reshapeSize,1]);
dG2_ii_73=reshape(dG1M.interictal(4).deltaG2,[reshapeSize,1]);
dG2_ii_159=reshape(dG1M.interictal(5).deltaG2,[reshapeSize,1]);
dG2_ii_239=reshape(dG1M.interictal(6).deltaG2,[reshapeSize,1]);
dG2_ii_358=reshape(dG1M.interictal(7).deltaG2,[reshapeSize,1]);
dG2_ii_479=reshape(dG1M.interictal(8).deltaG2,[reshapeSize,1]);
dG2_ii_498=reshape(dG1M.interictal(9).deltaG2,[reshapeSize,1]);
dG2_ii_500=reshape(dG1M.interictal(10).deltaG2,[reshapeSize,1]);

dG2_pi_1=reshape(dG1M.preictal(1).deltaG2,[reshapeSize,1]);
dG2_pi_5=reshape(dG1M.preictal(2).deltaG2,[reshapeSize,1]);
dG2_pi_10=reshape(dG1M.preictal(3).deltaG2,[reshapeSize,1]);
dG2_pi_15=reshape(dG1M.preictal(4).deltaG2,[reshapeSize,1]);
dG2_pi_20=reshape(dG1M.preictal(5).deltaG2,[reshapeSize,1]);
dG2_pi_25=reshape(dG1M.preictal(6).deltaG2,[reshapeSize,1]);
dG2_pi_30=reshape(dG1M.preictal(7).deltaG2,[reshapeSize,1]);
dG2_pi_35=reshape(dG1M.preictal(8).deltaG2,[reshapeSize,1]);
dG2_pi_40=reshape(dG1M.preictal(9).deltaG2,[reshapeSize,1]);
dG2_pi_42=reshape(dG1M.preictal(10).deltaG2,[reshapeSize,1]);

dG2ii_10k=[dG2_ii_1;dG2_ii_10;dG2_ii_20;dG2_ii_73;dG2_ii_159;dG2_ii_239;dG2_ii_358;dG2_ii_479;dG2_ii_498;dG2_ii_500];
dG2pi_10k=[dG2_pi_1;dG2_pi_5;dG2_pi_10;dG2_pi_15;dG2_pi_20;dG2_pi_25;dG2_pi_30;dG2_pi_35;dG2_pi_40;dG2_pi_42];

dG1_ii_1=reshape(dG15M.interictal(1).deltaG1,[reshapeSize,1]);
dG1_ii_10=reshape(dG15M.interictal(2).deltaG1,[reshapeSize,1]);
dG1_ii_20=reshape(dG15M.interictal(3).deltaG1,[reshapeSize,1]);
dG1_ii_73=reshape(dG15M.interictal(4).deltaG1,[reshapeSize,1]);
dG1_ii_159=reshape(dG15M.interictal(5).deltaG1,[reshapeSize,1]);
dG1_ii_239=reshape(dG15M.interictal(6).deltaG1,[reshapeSize,1]);
dG1_ii_358=reshape(dG15M.interictal(7).deltaG1,[reshapeSize,1]);
dG1_ii_479=reshape(dG15M.interictal(8).deltaG1,[reshapeSize,1]);
dG1_ii_498=reshape(dG15M.interictal(9).deltaG1,[reshapeSize,1]);
dG1_ii_500=reshape(dG15M.interictal(10).deltaG1,[reshapeSize,1]);

dG1_pi_1=reshape(dG15M.preictal(1).deltaG1,[reshapeSize,1]);
dG1_pi_5=reshape(dG15M.preictal(2).deltaG1,[reshapeSize,1]);
dG1_pi_10=reshape(dG15M.preictal(3).deltaG1,[reshapeSize,1]);
dG1_pi_15=reshape(dG15M.preictal(4).deltaG1,[reshapeSize,1]);
dG1_pi_20=reshape(dG15M.preictal(5).deltaG1,[reshapeSize,1]);
dG1_pi_25=reshape(dG15M.preictal(6).deltaG1,[reshapeSize,1]);
dG1_pi_30=reshape(dG15M.preictal(7).deltaG1,[reshapeSize,1]);
dG1_pi_35=reshape(dG15M.preictal(8).deltaG1,[reshapeSize,1]);
dG1_pi_40=reshape(dG15M.preictal(9).deltaG1,[reshapeSize,1]);
dG1_pi_42=reshape(dG15M.preictal(10).deltaG1,[reshapeSize,1]);

dG1ii_15k=[dG1_ii_1;dG1_ii_10;dG1_ii_20;dG1_ii_73;dG1_ii_159;dG1_ii_239;dG1_ii_358;dG1_ii_479;dG1_ii_498;dG1_ii_500];
dG1pi_15k=[dG1_pi_1;dG1_pi_5;dG1_pi_10;dG1_pi_15;dG1_pi_20;dG1_pi_25;dG1_pi_30;dG1_pi_35;dG1_pi_40;dG1_pi_42];

dG2_ii_1=reshape(dG15M.interictal(1).deltaG2,[reshapeSize,1]);
dG2_ii_10=reshape(dG15M.interictal(2).deltaG2,[reshapeSize,1]);
dG2_ii_20=reshape(dG15M.interictal(3).deltaG2,[reshapeSize,1]);
dG2_ii_73=reshape(dG15M.interictal(4).deltaG2,[reshapeSize,1]);
dG2_ii_159=reshape(dG15M.interictal(5).deltaG2,[reshapeSize,1]);
dG2_ii_239=reshape(dG15M.interictal(6).deltaG2,[reshapeSize,1]);
dG2_ii_358=reshape(dG15M.interictal(7).deltaG2,[reshapeSize,1]);
dG2_ii_479=reshape(dG15M.interictal(8).deltaG2,[reshapeSize,1]);
dG2_ii_498=reshape(dG15M.interictal(9).deltaG2,[reshapeSize,1]);
dG2_ii_500=reshape(dG15M.interictal(10).deltaG2,[reshapeSize,1]);

dG2_pi_1=reshape(dG15M.preictal(1).deltaG2,[reshapeSize,1]);
dG2_pi_5=reshape(dG15M.preictal(2).deltaG2,[reshapeSize,1]);
dG2_pi_10=reshape(dG15M.preictal(3).deltaG2,[reshapeSize,1]);
dG2_pi_15=reshape(dG15M.preictal(4).deltaG2,[reshapeSize,1]);
dG2_pi_20=reshape(dG15M.preictal(5).deltaG2,[reshapeSize,1]);
dG2_pi_25=reshape(dG15M.preictal(6).deltaG2,[reshapeSize,1]);
dG2_pi_30=reshape(dG15M.preictal(7).deltaG2,[reshapeSize,1]);
dG2_pi_35=reshape(dG15M.preictal(8).deltaG2,[reshapeSize,1]);
dG2_pi_40=reshape(dG15M.preictal(9).deltaG2,[reshapeSize,1]);
dG2_pi_42=reshape(dG15M.preictal(10).deltaG2,[reshapeSize,1]);

dG2ii_15k=[dG2_ii_1;dG2_ii_10;dG2_ii_20;dG2_ii_73;dG2_ii_159;dG2_ii_239;dG2_ii_358;dG2_ii_479;dG2_ii_498;dG2_ii_500];
dG2pi_15k=[dG2_pi_1;dG2_pi_5;dG2_pi_10;dG2_pi_15;dG2_pi_20;dG2_pi_25;dG2_pi_30;dG2_pi_35;dG2_pi_40;dG2_pi_42];

dG1_ii_1=reshape(dG2M.interictal(1).deltaG1,[reshapeSize,1]);
dG1_ii_10=reshape(dG2M.interictal(2).deltaG1,[reshapeSize,1]);
dG1_ii_20=reshape(dG2M.interictal(3).deltaG1,[reshapeSize,1]);
dG1_ii_73=reshape(dG2M.interictal(4).deltaG1,[reshapeSize,1]);
dG1_ii_159=reshape(dG2M.interictal(5).deltaG1,[reshapeSize,1]);
dG1_ii_239=reshape(dG2M.interictal(6).deltaG1,[reshapeSize,1]);
dG1_ii_358=reshape(dG2M.interictal(7).deltaG1,[reshapeSize,1]);
dG1_ii_479=reshape(dG2M.interictal(8).deltaG1,[reshapeSize,1]);
dG1_ii_498=reshape(dG2M.interictal(9).deltaG1,[reshapeSize,1]);
dG1_ii_500=reshape(dG2M.interictal(10).deltaG1,[reshapeSize,1]);

dG1_pi_1=reshape(dG2M.preictal(1).deltaG1,[reshapeSize,1]);
dG1_pi_5=reshape(dG2M.preictal(2).deltaG1,[reshapeSize,1]);
dG1_pi_10=reshape(dG2M.preictal(3).deltaG1,[reshapeSize,1]);
dG1_pi_15=reshape(dG2M.preictal(4).deltaG1,[reshapeSize,1]);
dG1_pi_20=reshape(dG2M.preictal(5).deltaG1,[reshapeSize,1]);
dG1_pi_25=reshape(dG2M.preictal(6).deltaG1,[reshapeSize,1]);
dG1_pi_30=reshape(dG2M.preictal(7).deltaG1,[reshapeSize,1]);
dG1_pi_35=reshape(dG2M.preictal(8).deltaG1,[reshapeSize,1]);
dG1_pi_40=reshape(dG2M.preictal(9).deltaG1,[reshapeSize,1]);
dG1_pi_42=reshape(dG2M.preictal(10).deltaG1,[reshapeSize,1]);

dG1ii_20k=[dG1_ii_1;dG1_ii_10;dG1_ii_20;dG1_ii_73;dG1_ii_159;dG1_ii_239;dG1_ii_358;dG1_ii_479;dG1_ii_498;dG1_ii_500];
dG1pi_20k=[dG1_pi_1;dG1_pi_5;dG1_pi_10;dG1_pi_15;dG1_pi_20;dG1_pi_25;dG1_pi_30;dG1_pi_35;dG1_pi_40;dG1_pi_42];

dG2_ii_1=reshape(dG2M.interictal(1).deltaG2,[reshapeSize,1]);
dG2_ii_10=reshape(dG2M.interictal(2).deltaG2,[reshapeSize,1]);
dG2_ii_20=reshape(dG2M.interictal(3).deltaG2,[reshapeSize,1]);
dG2_ii_73=reshape(dG2M.interictal(4).deltaG2,[reshapeSize,1]);
dG2_ii_159=reshape(dG2M.interictal(5).deltaG2,[reshapeSize,1]);
dG2_ii_239=reshape(dG2M.interictal(6).deltaG2,[reshapeSize,1]);
dG2_ii_358=reshape(dG2M.interictal(7).deltaG2,[reshapeSize,1]);
dG2_ii_479=reshape(dG2M.interictal(8).deltaG2,[reshapeSize,1]);
dG2_ii_498=reshape(dG2M.interictal(9).deltaG2,[reshapeSize,1]);
dG2_ii_500=reshape(dG2M.interictal(10).deltaG2,[reshapeSize,1]);

dG2_pi_1=reshape(dG2M.preictal(1).deltaG2,[reshapeSize,1]);
dG2_pi_5=reshape(dG2M.preictal(2).deltaG2,[reshapeSize,1]);
dG2_pi_10=reshape(dG2M.preictal(3).deltaG2,[reshapeSize,1]);
dG2_pi_15=reshape(dG2M.preictal(4).deltaG2,[reshapeSize,1]);
dG2_pi_20=reshape(dG2M.preictal(5).deltaG2,[reshapeSize,1]);
dG2_pi_25=reshape(dG2M.preictal(6).deltaG2,[reshapeSize,1]);
dG2_pi_30=reshape(dG2M.preictal(7).deltaG2,[reshapeSize,1]);
dG2_pi_35=reshape(dG2M.preictal(8).deltaG2,[reshapeSize,1]);
dG2_pi_40=reshape(dG2M.preictal(9).deltaG2,[reshapeSize,1]);
dG2_pi_42=reshape(dG2M.preictal(10).deltaG2,[reshapeSize,1]);

dG2ii_20k=[dG2_ii_1;dG2_ii_10;dG2_ii_20;dG2_ii_73;dG2_ii_159;dG2_ii_239;dG2_ii_358;dG2_ii_479;dG2_ii_498;dG2_ii_500];
dG2pi_20k=[dG2_pi_1;dG2_pi_5;dG2_pi_10;dG2_pi_15;dG2_pi_20;dG2_pi_25;dG2_pi_30;dG2_pi_35;dG2_pi_40;dG2_pi_42];

dG1=[dG1ii_10k,dG1ii_15k,dG1ii_20k;dG1pi_10k,dG1pi_15k,dG1pi_20k];
dG2=[dG2ii_10k,dG2ii_15k,dG2ii_20k;dG2pi_10k,dG2pi_15k,dG2pi_20k];

classTypes=zeros(size(dG1,1),1);

for i=1:length(classTypes)
    if i<length(classTypes)/2+1
        classTypes(i)=1; 
    else
        classTypes(i)=2;
    end
end

%Support Vector Machine
%'KernelFunction','gaussian'
Mdl1=fitcsvm(dG1,classTypes,'KernelFunction','gaussian','Standardize',1);
classOrder = Mdl1.ClassNames;
sv = Mdl1.SupportVectors;

figure(1);
sgtitle('Definition 1 of dG','FontWeight','bold','Fontsize',18);
subplot(1,3,1);
gscatter(dG1(:,1),dG1(:,2),classTypes);
title('dG ({\itA} = 1*10^6) vs dG ({\itA} = 1.5*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 1.5*10^6) (S)','Fontsize',14);
legend('Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
gscatter(dG1(:,1),dG1(:,3),classTypes);
title('dG ({\itA} = 1*10^6) vs dG ({\itA} = 2*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 2*10^6) (S)','Fontsize',14);
legend('Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
gscatter(dG1(:,2),dG1(:,3),classTypes);
title('dG ({\itA} = 1.5*10^6) vs dG ({\itA} = 2*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1.5*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 2*10^6) (S)','Fontsize',14);
% hold on
%plot(sv(:,1),sv(:,3),'ko','MarkerSize',10)
legend('Interictal','Preictal','Fontsize',14);

figure(2);
sgtitle('Definition 2 of dG','FontWeight','bold','Fontsize',18);
subplot(1,3,1);
gscatter(dG2(:,1),dG2(:,2),classTypes);
title('dG ({\itA} = 1*10^6) vs dG ({\itA} = 1.5*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 1.5*10^6) (S)','Fontsize',14);
legend('Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
gscatter(dG2(:,1),dG2(:,3),classTypes);
title('dG ({\itA} = 1*10^6) vs dG ({\itA} = 2*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 2*10^6) (S)','Fontsize',14);
legend('Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
gscatter(dG2(:,2),dG2(:,3),classTypes);
title('dG ({\itA} = 1.5*10^6) vs dG ({\itA} = 2*10^6)','Fontsize',15);
xlabel('dG ({\itA} = 1.5*10^6) (S)','Fontsize',14),ylabel('dG ({\itA} = 2*10^6) (S)','Fontsize',14);
% hold on
%plot(sv(:,1),sv(:,3),'ko','MarkerSize',10)
legend('Interictal','Preictal','Fontsize',14);

figure(3);
scatter3(dG1(1:10240,1),dG1(1:10240,2),dG1(1:10240,3),'.r');
hold on;
scatter3(dG1(10241:end,1),dG1(10241:end,2),dG1(10241:end,3),'.b');
title('3D Scatter Plot of dG (Definition 1)','Fontsize',18);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',15),ylabel('{\itA} = dG (1.5*10^6) (S)','Fontsize',15),zlabel('dG ({\itA} = 2*10^6) (S)','Fontsize',15);
legend('Interictal','Preictal','Fontsize',15);

figure(4);
scatter3(dG2(1:10240,1),dG2(1:10240,2),dG2(1:10240,3),'.r');
hold on;
scatter3(dG2(10241:end,1),dG2(10241:end,2),dG2(10241:end,3),'.b');
title('3D Scatter Plot of dG (Definition 2)','Fontsize',18);
xlabel('dG ({\itA} = 1*10^6) (S)','Fontsize',15),ylabel('dG ({\itA} = 1.5*10^6) (S)','Fontsize',15),zlabel('dG ({\itA} = 2*10^6) (S)','Fontsize',15);
legend('Interictal','Preictal','Fontsize',15);

tabulate(classTypes);
Mdl2=fitcnb(dG1(:,1:2),classTypes,'ClassNames',{'1','2'});

figure(5)
gscatter(dG1(:,1),dG1(:,2),classTypes);
h = gca;
cxlim = h.XLim;
cylim = h.YLim;
hold on
Params = cell2mat(Mdl2.DistributionParameters); 
Mu = Params(2*(1:2)-1,1:2); % Extract the means
Sigma = zeros(2,2,2);
for j = 1:2
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
    ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
    f = @(x,y) arrayfun(@(x0,y0) mvnpdf([x0 y0],Mu(j,:),Sigma(:,:,j)),x,y);
    fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier')
xlabel('20k')
ylabel('10k')
legend('Interictal','Preictal')
hold off

%% Trying LDA
Mdl3=fitcdiscr(dG1,classTypes);
SbSwINV=Mdl3.BetweenSigma\(Mdl3.Sigma);
[eigenVectors,eigenValues]=eig(SbSwINV);
[~,I]=sort(diag(eigenValues),"descend");

featureVectors=eigenVectors(:,I); 
interictalData=dG1(1:size(dG1,1)/2,:);
preictalData=dG12(size(dG1,1)/2+1:end,:);
projectedInterictal=interictalData*featureVectors(:,1:2);
projectedPreictal=preictalData*featureVectors(:,1:2);

figure(4);
plot(projectedInterictal(:,1),projectedInterictal(:,2),strcat('ob'),projectedPreictal(:,1),projectedPreictal(:,2),strcat('or'),'MarkerSize',10);
hold on;
legend('Interictal','Preictal');