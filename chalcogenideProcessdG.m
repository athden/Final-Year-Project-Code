%% Chalcogenide Model: Processing the dG values & Correlation matrix
clear,clc,close all;

%Loading the file with the dG heatmaps for each segment and amplification gain
load 'chalcogenidedGVal.mat';

%Selecting one of the structs in chalcogenidedGVal.mat for analysis
dG=dG1M;

%Initialising the ensemble-averaged heatmaps
dG1_ii_ensemble=zeros(16,64);
dG2_ii_ensemble=zeros(16,64);
dG1_pi_ensemble=zeros(16,64);
dG2_pi_ensemble=zeros(16,64);

%Plotting the heatmaps for the interictal data
for seg_ii=1:size(dG.interictal,2)
    %Plotting definition 1 of the dG
    figure(seg_ii),subplot(2,1,1);
    h=heatmap(dG.interictal(seg_ii).deltaG1,'XLabel','Segment number{\it n}','YLabel','Channel','Title',{['Heatmap for interictal segment ',num2str(dG.interictal(seg_ii).Segment)] ['Definition 1']},'Fontsize',15);
    h.Colormap=autumn(256);
    dG1_ii_ensemble=dG1_ii_ensemble+dG.interictal(seg_ii).deltaG1;
    
    %Plotting definition 2 of the dG
    subplot(2,1,2);
    h=heatmap(dG.interictal(seg_ii).deltaG2,'XLabel','Segment number{\it n}','YLabel','Channel','Title',{['Heatmap for interictal segment ',num2str(dG.interictal(seg_ii).Segment)] ['Definition 2']},'Fontsize',15);
    h.Colormap=autumn(256);
    dG2_ii_ensemble=dG2_ii_ensemble+dG.interictal(seg_ii).deltaG2;
end
%Ensemble averaged heatmap from the interictal segments
dG1_ii_ensemble=dG1_ii_ensemble./size(dG.interictal,2);
dG2_ii_ensemble=dG2_ii_ensemble./size(dG.interictal,2);

%Plotting the heatmaps for the preictal data
for seg_pi=1:size(dG.preictal,2)
    %Plotting definition 1 of the dG
    figure(seg_pi+size(dG.interictal,2)),subplot(2,1,1);
    h=heatmap(dG.preictal(seg_pi).deltaG1,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Heatmap for preictal segment ',num2str(dG.preictal(seg_pi).Segment)] ['Definition 1']},'Fontsize',15);
    h.Colormap=autumn(256);
    dG1_pi_ensemble=dG1_pi_ensemble+dG.preictal(seg_pi).deltaG1;
    
    %Plotting definition 2 of the dG
    subplot(2,1,2);
    h=heatmap(dG.preictal(seg_pi).deltaG2,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Heatmap for preictal segment ',num2str(dG.preictal(seg_pi).Segment)] ['Definition 2']},'Fontsize',15);
    h.Colormap=autumn(256);
    dG2_pi_ensemble=dG2_pi_ensemble+dG.preictal(seg_pi).deltaG2;
end
%Ensemble averaged heatmap from the interictal segments
dG1_pi_ensemble=dG1_pi_ensemble./size(dG.preictal,2);
dG2_pi_ensemble=dG2_pi_ensemble./size(dG.preictal,2);

%Figure indexing
idx=seg_ii+seg_pi+1;
figure(idx);
%Plotting the ensemble averaged heatmaps
subplot(2,2,1);
h=heatmap(dG1_ii_ensemble,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Ensemble-averaged heatmap for interictal segments'] ['Gain = 1*10^6 in-parallel (Definition 1)']},'Fontsize',15);
h.Colormap=autumn(256);
subplot(2,2,2);
h=heatmap(dG2_ii_ensemble,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Ensemble-averaged heatmap for interictal segments'] ['Gain = 1*10^6 in-parallel (Definition 2)']},'Fontsize',15);
h.Colormap=autumn(256);
subplot(2,2,3);
h=heatmap(dG1_pi_ensemble,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Ensemble-averaged heatmap for preictal segments'] ['Gain = 1*10^6 in-parallel (Definition 1)']},'Fontsize',15);
h.Colormap=autumn(256);
subplot(2,2,4);
h=heatmap(dG2_pi_ensemble,'XLabel','Column number{\it n}','YLabel','Channel','Title',{['Ensemble-averaged heatmap for preictal segments'] ['Gain = 1*10^6 in-parallel (Definition 2)']},'Fontsize',15);
h.Colormap=autumn(256);

%Plotting the histogram of the ensemble averaged heatmap for the interictal data
figure(idx+1),subplot(2,2,1);
histogram(dG1_ii_ensemble*10^6,50),hold on;
title({['Histogram for the ensemble {\DeltaG} of ',num2str(size(dG.interictal,2)),' interictal segments'] ['Gain = 1*10^6 in-parallel (Definition 1)']},'Fontsize',17);
xlabel('{\DeltaG} ({\muS})','Fontsize',16);
ylabel('Counts','Fontsize',16);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_ii=mean(dG1_ii_ensemble*10^6,'all');
std_ensemble_ii=std(dG1_ii_ensemble*10^6,0,'all');
% std_ensemble_ii=mean(std_ensemble_ii); %POTENTIALLY WRONG
xline(mean_ensemble_ii,'Color','k','LineWidth',2);
xline(mean_ensemble_ii-std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_ii+std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_ii),'{\mu}S'],['Standard deviation = ',num2str(std_ensemble_ii),'{\mu}S'],'Fontsize',13);

subplot(2,2,2);
histogram(dG2_ii_ensemble*10^6,50),hold on;
title({['Histogram for the ensemble {\DeltaG} of ',num2str(size(dG.interictal,2)),' interictal segments'] ['Gain = 1*10^6 in-parallel (Definition 2)']},'Fontsize',17);
xlabel('{\DeltaG} ({\muS})','Fontsize',16);
ylabel('Counts','Fontsize',16);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_ii=mean(dG2_ii_ensemble*10^6,'all');
std_ensemble_ii=std(dG2_ii_ensemble*10^6,0,'all');
%std_ensemble_ii=mean(std_ensemble_ii);
xline(mean_ensemble_ii,'Color','k','LineWidth',2);
xline(mean_ensemble_ii-std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_ii+std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_ii),'{\mu}S'],['Standard deviation = ',num2str(std_ensemble_ii),'{\mu}S'],'Fontsize',13);

subplot(2,2,3);
hp=histogram(dG1_pi_ensemble*10^6,50);
hold on;
hp.FaceColor='r';
title({['Histogram for the ensemble {\DeltaG} of ',num2str(size(dG.preictal,2)),' preictal segments'] ['Gain = 1*10^6 in-parallel (Definition 1)']},'Fontsize',17);
xlabel('{\DeltaG} ({\muS})','Fontsize',16);
ylabel('Counts','Fontsize',16);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_pi=mean(dG1_pi_ensemble*10^6,'all');
std_ensemble_pi=std(dG1_pi_ensemble*10^6,0,'all');
%std_ensemble_pi=mean(std_ensemble_pi);
xline(mean_ensemble_pi,'Color','k','LineWidth',2);
xline(mean_ensemble_pi-std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_pi+std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_pi),'{\mu}S'],['Standard deviation = ',num2str(std_ensemble_pi),'{\mu}S'],'Fontsize',13);

subplot(2,2,4);
hp=histogram(dG2_pi_ensemble*10^6,50);
hold on;
hp.FaceColor='r';
title({['Histogram for the ensemble {\DeltaG} of ',num2str(size(dG.preictal,2)),' preictal segments'] ['Gain = 1*10^6 in-parallel (Definition 2)']},'Fontsize',17);
xlabel('{\DeltaG} ({\muS})','Fontsize',13);
ylabel('Counts','Fontsize',13);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_pi=mean(dG2_pi_ensemble*10^6,'all');
std_ensemble_pi=std(dG2_pi_ensemble*10^6,0,'all');
%std_ensemble_pi=mean(std_ensemble_pi);
xline(mean_ensemble_pi,'Color','k','LineWidth',2);
xline(mean_ensemble_pi-std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_pi+std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_pi),'{\mu}S'],['Standard deviation = ',num2str(std_ensemble_pi),'{\mu}S'],'Fontsize',13);

%Robustness Analysis
for i1=1:seg_ii
    %Computing the correlation coefficient between the dG matrices between each interictal segment
    for i2=1:seg_ii
        %Interictal-interictal correlation
        R_ii=corrcoef(dG.interictal(i1).deltaG1,dG.interictal(i2).deltaG1); %definition 1
        R_ii2=corrcoef(dG.interictal(i1).deltaG2,dG.interictal(i2).deltaG2); %definition 2
        %2x2 matrix with ones in diagonal and correlation coeffs in off-diagonal
        r_ii(i1,i2)=R_ii(1,2); %Correlation coefficients using definition 1
        r_ii2(i1,i2)=R_ii2(1,2); %Correlation coefficients using definition 1
    end
    
    %Interictal-ensemble interictal correlation
    R_ii_ensi=corrcoef(dG1_ii_ensemble,dG.interictal(i1).deltaG1);
    r_ii_ensi(i1)=R_ii_ensi(1,2); %Correlation coefficients using definition 1
    R_ii_ensi2=corrcoef(dG2_ii_ensemble,dG.interictal(i1).deltaG2);
    r_ii_ensi2(i1)=R_ii_ensi2(1,2); %Correlation coefficients using definition 1
    
    %Interictal-ensemble preictal correlation
    R_ip_ensp=corrcoef(dG.interictal(i1).deltaG1,dG1_pi_ensemble);
    r_ip_ensp(i1)=R_ip_ensp(1,2); %Correlation coefficients using definition 1
    R_ip_ensp2=corrcoef(dG.interictal(i1).deltaG2,dG2_pi_ensemble);
    r_ip_ensp2(i1)=R_ip_ensp2(1,2); %Correlation coefficients using definition 1
    
    for p1=1:seg_pi
        %Interictal-preictal correlation
        R_ip=corrcoef(dG.interictal(i1).deltaG1,dG.preictal(p1).deltaG1);
        r_ip(i1,p1)=R_ip(1,2);
        R_ip2=corrcoef(dG.interictal(i1).deltaG2,dG.preictal(p1).deltaG2);
        r_ip2(i1,p1)=R_ip2(1,2);
        
        %Ensemble interictal-preictal correlation
        R_ip_ensi=corrcoef(dG.preictal(p1).deltaG1,dG1_ii_ensemble);
        r_ip_ensi(p1)=R_ip_ensi(1,2); %Correlation coefficients
        R_ip_ensi2=corrcoef(dG.preictal(p1).deltaG2,dG2_ii_ensemble);
        r_ip_ensi2(p1)=R_ip_ensi2(1,2); %Correlation coefficients
        
        for p2=1:seg_pi
            %Preictal-preictal correlation
            R_pp=corrcoef(dG.preictal(p1).deltaG1,dG.preictal(p2).deltaG1);
            r_pp(p1,p2)=R_pp(1,2);
            R_pp2=corrcoef(dG.preictal(p1).deltaG2,dG.preictal(p2).deltaG2);
            r_pp2(p1,p2)=R_pp2(1,2);
        end
        
        %Preictal-ensemble preictal correlation
        R_pp_ens=corrcoef(dG.preictal(p1).deltaG1,dG1_pi_ensemble);
        r_pp_ens(p1)=R_pp_ens(1,2);
        R_pp_ens2=corrcoef(dG.preictal(p1).deltaG2,dG2_pi_ensemble);
        r_pp_ens2(p1)=R_pp_ens2(1,2);
    end
end

%Plotting the correlation matrix of the different segments
CorrMat=[r_ii,r_ip;r_ip',r_pp];
figure(idx+2),subplot(2,2,1);
map=redblue(100);
hcorr1=heatmap(CorrMat,'XLabel','Segment number{\it n}','YLabel','Segment number{\it n}','Title',{['Correlation matrix of output {\DeltaG}'] ['Gain = 1*10^6 (Definition 1)']},'Fontsize',15);
hcorr1.Colormap=map;

%Creating a Toeplitz symmetrical matrix using the correlation coefficients wrt the ensemble
r_ii_e=toeplitz(r_ii_ensi);
r_pp_e=toeplitz(r_pp_ens);

%Plotting the correlation matrix of the different segments
%Consider using clim for the colourbar
CorrMat=[r_ii_e,r_ip;r_ip',r_pp_e];
subplot(2,2,2);
hcorr2=heatmap(CorrMat,'XLabel','Segment number{\it n}','YLabel','Segment number{\it n}','Title',{['Correlation matrix of the output {\DeltaG}'] ['using ensemble-averaged heatmaps'] ['Gain = 1*10^6 (Definition 1)']},'Fontsize',15);
hcorr2.Colormap=map;

%Plotting the correlation matrix of the different segments
CorrMat2=[r_ii2,r_ip2;r_ip2',r_pp2];
subplot(2,2,3);
map=redblue(100);
hcorr1=heatmap(CorrMat2,'XLabel','Segment number{\it n}','YLabel','Segment number{\it n}','Title',{['Correlation matrix of the output {\DeltaG}'] ['Gain = 1*10^6 (Definition 2)']},'Fontsize',15);
hcorr1.Colormap=map;

%Creating a Toeplitz symmetrical matrix using the correlation coefficients wrt the ensemble
r_ii_e2=toeplitz(r_ii_ensi2);
r_pp_e2=toeplitz(r_pp_ens2);

%Plotting the correlation matrix of the different segments
%Consider using clim for the colourbar
CorrMat2=[r_ii_e2,r_ip;r_ip',r_pp_e2];
subplot(2,2,4);
hcorr2=heatmap(CorrMat2,'XLabel','Segment number{\it n}','YLabel','Segment number{\it n}','Title',{['Correlation matrix of the output {\DeltaG}']  ['using ensemble-averaged heatmaps'] ['Gain = 1*10^6 (Definition 2)']},'Fontsize',15);
hcorr2.Colormap=map;

% Testing an extra 5 interictal and preictal segments
%Assigning a class to the new set of 5 interictal and preictal signals
load 'extrachalcogenidedGVal.mat';

%Selecting one of the amplification gains and selecting a heatmap from
%there for comparison against the ensemble averages

%Comparison of selected heatmap against interictal ensembled-average
R_ensi_sample=corrcoef(extradG1M.interictal(1).deltaG1,dG1_ii_ensemble);
r_ensi_sample=R_ensi_sample(1,2);

%Comparison of selected heatmap against interictal ensembled-average
R_ensp_sample=corrcoef(extradG1M.interictal(1).deltaG1,dG1_pi_ensemble);
r_ensp_sample=R_ensp_sample(1,2);

%% Function for the colourmap
%Source: https://uk.mathworks.com/matlabcentral/fileexchange/-red-blue-colormap

function c = redblue(m)
%REDBLUE    Shades of red and blue color map
%   REDBLUE(M), is an M-by-3 matrix that defines a colormap.
%   The colors begin with bright blue, range through shades of
%   blue to white, and then through shades of red to bright red.
%   REDBLUE, by itself, is the same length as the current figure's
%   colormap. If no figure exists, MATLAB creates one.
%   See also HSV, GRAY, HOT, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.

%   Adam Auton, 9th October 2009

    if nargin < 1, m = size(get(gcf,'colormap'),1); end

    if (mod(m,2) == 0)
        % From [0 0 1] to [1 1 1], then [1 1 1] to [1 0 0];
        m1 = m*0.5;
        r = (0:m1-1)'/max(m1-1,1);
        g = r;
        r = [r; ones(m1,1)];
        g = [g; flipud(g)];
        b = flipud(r);
    else
        % From [0 0 1] to [1 1 1] to [1 0 0];
        m1 = floor(m*0.5);
        r = (0:m1-1)'/max(m1,1);
        g = r;
        r = [r; ones(m1+1,1)];
        g = [g; 1; flipud(g)];
        b = flipud(r);
    end
    c = [r g b];
end
