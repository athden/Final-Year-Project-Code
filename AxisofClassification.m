%% Attempting to differentiate based on the signal energy and variation
clear,clc,close all;

load 'chalcogenidedGVal.mat';

for i=1:10
    ii_m1_1M(i,:)=mean(dG1M.interictal(i).deltaG1);
    ii_m2_1M(i,:)=mean(dG1M.interictal(i).deltaG2);
    pi_m1_1M(i,:)=mean(dG1M.preictal(i).deltaG1);
    pi_m2_1M(i,:)=mean(dG1M.preictal(i).deltaG2);
    
    ii_s1_1M(i,:)=std(dG1M.interictal(i).deltaG1);
    ii_s2_1M(i,:)=std(dG1M.interictal(i).deltaG2);
    pi_s1_1M(i,:)=std(dG1M.preictal(i).deltaG1);
    pi_s2_1M(i,:)=std(dG1M.preictal(i).deltaG2);
    
    ii_m1_15M(i,:)=mean(dG15M.interictal(i).deltaG1);
    ii_m2_15M(i,:)=mean(dG15M.interictal(i).deltaG2);
    pi_m1_15M(i,:)=mean(dG15M.preictal(i).deltaG1);
    pi_m2_15M(i,:)=mean(dG15M.preictal(i).deltaG2);
    
    ii_s1_15M(i,:)=std(dG15M.interictal(i).deltaG1);
    ii_s2_15M(i,:)=std(dG15M.interictal(i).deltaG2);
    pi_s1_15M(i,:)=std(dG15M.preictal(i).deltaG1);
    pi_s2_15M(i,:)=std(dG15M.preictal(i).deltaG2);
    
    ii_m1_2M(i,:)=mean(dG2M.interictal(i).deltaG1);
    ii_m2_2M(i,:)=mean(dG2M.interictal(i).deltaG2);
    pi_m1_2M(i,:)=mean(dG2M.preictal(i).deltaG1);
    pi_m2_2M(i,:)=mean(dG2M.preictal(i).deltaG2);
    
    ii_s1_20M(i,:)=std(dG2M.interictal(i).deltaG1);
    ii_s2_20M(i,:)=std(dG2M.interictal(i).deltaG2);
    pi_s1_20M(i,:)=std(dG2M.preictal(i).deltaG1);
    pi_s2_20M(i,:)=std(dG2M.preictal(i).deltaG2);
end

figure(1);
sgtitle('Average {\DeltaG} against Variability (Definition 1) - Chalcogenide-based Model','Fontweight','bold','Fontsize',18);
subplot(1,3,1);
p(1)=plot(ii_s1_1M(1,1),ii_m1_1M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_1M,ii_m1_1M,'.b');
p(2)=plot(pi_s1_1M(1,1),pi_m1_1M(1,1),'.r'); %for legend
plot(pi_s1_1M,pi_m1_1M,'.r');
title('{\itA} = 1*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
p(1)=plot(ii_s1_15M(1,1),ii_m1_15M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_15M,ii_m1_15M,'.b');
p(2)=plot(pi_s1_15M(1,1),pi_m1_15M(1,1),'.r'); %for legend
plot(pi_s1_15M,pi_m1_15M,'.r');
title('{\itA} = 1.5*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
p(1)=plot(ii_s1_20M(1,1),ii_m1_2M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_20M,ii_m1_2M,'.b');
p(2)=plot(pi_s1_20M(1,1),pi_m1_2M(1,1),'.r'); %for legend
plot(pi_s1_20M,pi_m1_2M,'.r');
title('{\itA} = 2*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);


figure(2);
sgtitle('Average {\DeltaG} against Variability (Definition 2) - Chalcogenide-based Model','Fontweight','bold','Fontsize',18);
subplot(1,3,1);
p(1)=plot(ii_s2_1M(1,1),ii_m2_1M(1,1),'.b'); %for legend
hold on;
plot(ii_s2_1M,ii_m2_1M,'.b');
p(2)=plot(pi_s2_1M(1,1),pi_m2_1M(1,1),'.r'); %for legend
plot(pi_s2_1M,pi_m2_1M,'.r');
title('{\itA} = 1*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
p(1)=plot(ii_s2_15M(1,1),ii_m2_15M(1,1),'.b'); %for legend
hold on;
plot(ii_s2_15M,ii_m2_15M,'.b');
p(2)=plot(pi_s2_15M(1,1),pi_m2_15M(1,1),'.r'); %for legend
plot(pi_s2_15M,pi_m2_15M,'.r');
title('{\itA} = 1.5*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
p(1)=plot(ii_s2_20M(1,1),ii_m2_2M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_20M,ii_m1_2M,'.b');
p(2)=plot(pi_s1_20M(1,1),pi_m1_2M(1,1),'.r'); %for legend
plot(pi_s1_20M,pi_m1_2M,'.r');
title('{\itA} = 2*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);

load 'HPdGVal.mat';

for i=1:10
    ii_m1_1M(i,:)=mean(abs(dG1M7nm.interictal(i).deltaG1));
    ii_m2_1M(i,:)=mean(abs(dG1M7nm.interictal(i).deltaG2));
    pi_m1_1M(i,:)=mean(abs(dG1M7nm.preictal(i).deltaG1));
    pi_m2_1M(i,:)=mean(abs(dG1M7nm.preictal(i).deltaG2));
    
    ii_s1_1M(i,:)=std(abs(dG1M7nm.interictal(i).deltaG1));
    ii_s2_1M(i,:)=std(abs(dG1M7nm.interictal(i).deltaG2));
    pi_s1_1M(i,:)=std(abs(dG1M7nm.preictal(i).deltaG1));
    pi_s2_1M(i,:)=std(abs(dG1M7nm.preictal(i).deltaG2));
    
    ii_m1_15M(i,:)=mean(abs(dG15M7nm.interictal(i).deltaG1));
    ii_m2_15M(i,:)=mean(abs(dG15M7nm.interictal(i).deltaG2));
    pi_m1_15M(i,:)=mean(abs(dG15M7nm.preictal(i).deltaG1));
    pi_m2_15M(i,:)=mean(abs(dG15M7nm.preictal(i).deltaG2));
    
    ii_s1_15M(i,:)=std(abs(dG15M7nm.interictal(i).deltaG1));
    ii_s2_15M(i,:)=std(abs(dG15M7nm.interictal(i).deltaG2));
    pi_s1_15M(i,:)=std(abs(dG15M7nm.preictal(i).deltaG1));
    pi_s2_15M(i,:)=std(abs(dG15M7nm.preictal(i).deltaG2));
    
    ii_m1_2M(i,:)=mean(abs(dG2M7nm.interictal(i).deltaG1));
    ii_m2_2M(i,:)=mean(abs(dG2M7nm.interictal(i).deltaG2));
    pi_m1_2M(i,:)=mean(abs(dG2M7nm.preictal(i).deltaG1));
    pi_m2_2M(i,:)=mean(abs(dG2M7nm.preictal(i).deltaG2));
    
    ii_s1_2M(i,:)=std(abs(dG2M7nm.interictal(i).deltaG1));
    ii_s2_2M(i,:)=std(abs(dG2M7nm.interictal(i).deltaG2));
    pi_s1_2M(i,:)=std(abs(dG2M7nm.preictal(i).deltaG1));
    pi_s2_2M(i,:)=std(abs(dG2M7nm.preictal(i).deltaG2));
end

figure(3);
sgtitle('Average {\DeltaG} against Variability (Definition 1) - HP Model','Fontweight','bold','Fontsize',18);
subplot(1,3,1);
p(1)=plot(ii_s1_1M(1,1),ii_m1_1M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_1M,ii_m1_1M,'.b');
p(2)=plot(pi_s1_1M(1,1),pi_m1_1M(1,1),'.r'); %for legend
plot(pi_s1_1M,pi_m1_1M,'.r');
title('{\itA} = 1*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
p(1)=plot(ii_s1_15M(1,1),ii_m1_15M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_15M,ii_m1_15M,'.b');
p(2)=plot(pi_s1_15M(1,1),pi_m1_15M(1,1),'.r'); %for legend
plot(pi_s1_15M,pi_m1_15M,'.r');
title('{\itA} = 1.5*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
p(1)=plot(ii_s1_20M(1,1),ii_m1_2M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_20M,ii_m1_2M,'.b');
p(2)=plot(pi_s1_20M(1,1),pi_m1_2M(1,1),'.r'); %for legend
plot(pi_s1_20M,pi_m1_2M,'.r');
title('{\itA} = 2*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);

figure(4);
sgtitle('Average {\DeltaG} against Variability (Definition 2) - HP Model','Fontweight','bold','Fontsize',18);
subplot(1,3,1);
p(1)=plot(ii_s2_1M(1,1),ii_m2_1M(1,1),'.b'); %for legend
hold on;
plot(ii_s2_1M,ii_m2_1M,'.b');
p(2)=plot(pi_s2_1M(1,1),pi_m2_1M(1,1),'.r'); %for legend
plot(pi_s2_1M,pi_m2_1M,'.r');
title('{\itA} = 1*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,2);
p(1)=plot(ii_s2_15M(1,1),ii_m2_15M(1,1),'.b'); %for legend
hold on;
plot(ii_s2_15M,ii_m2_15M,'.b');
p(2)=plot(pi_s2_15M(1,1),pi_m2_15M(1,1),'.r'); %for legend
plot(pi_s2_15M,pi_m2_15M,'.r');
title('{\itA} = 1.5*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);
subplot(1,3,3);
p(1)=plot(ii_s2_20M(1,1),ii_m2_2M(1,1),'.b'); %for legend
hold on;
plot(ii_s1_20M,ii_m1_2M,'.b');
p(2)=plot(pi_s1_20M(1,1),pi_m1_2M(1,1),'.r'); %for legend
plot(pi_s1_20M,pi_m1_2M,'.r');
title('{\itA} = 2*10^6','Fontsize',15);
xlabel('Variation (S)','Fontsize',14),ylabel('Average {\DeltaG} (S)','Fontsize',14);
legend([p(1) p(2)],'Interictal','Preictal','Fontsize',14);


