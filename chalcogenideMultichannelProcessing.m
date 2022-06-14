%Multichannel parallel processing of neural signals
clear,clc,close all;
tic

%Loading the interictal data (10 random segments)
load 'Dog_2_interictal_segment_0001.mat';
load 'Dog_2_interictal_segment_0010.mat';
load 'Dog_2_interictal_segment_0020.mat';
load 'Dog_2_interictal_segment_0073.mat';
load 'Dog_2_interictal_segment_0159.mat';
load 'Dog_2_interictal_segment_0239.mat';
load 'Dog_2_interictal_segment_0358.mat';
load 'Dog_2_interictal_segment_0479.mat';
load 'Dog_2_interictal_segment_0498.mat';
load 'Dog_2_interictal_segment_0500.mat';

unit=10^(-6);

%Data comes from iEEG which should theoretically be in units of μV, therefore dividing by 10^6
interictal_segment_1.data=interictal_segment_1.data.*unit; %sequence 1
interictal_segment_10.data=interictal_segment_10.data.*unit; %sequence 4
interictal_segment_20.data=interictal_segment_20.data.*unit; %sequence 2
interictal_segment_73.data=interictal_segment_73.data.*unit; %sequence 1
interictal_segment_159.data=interictal_segment_159.data.*unit; %sequence 3
interictal_segment_239.data=interictal_segment_239.data.*unit; %sequence 5
interictal_segment_358.data=interictal_segment_358.data.*unit; %sequence 4
interictal_segment_479.data=interictal_segment_479.data.*unit; %sequence 5
interictal_segment_498.data=interictal_segment_498.data.*unit; %sequence 6
interictal_segment_500.data=interictal_segment_500.data.*unit; %sequence 2

%Obtaining the time axis given the specific sampling frequency
fs=round(interictal_segment_1.sampling_frequency);
time=[0:1/fs:(length(interictal_segment_500.data)-1)/fs];

%Plotting the raw signal for segment 1 to visualise
figure(1),subplot(2,1,1);
%Plotting all 16 channels of segment 1
for channel=1:size(interictal_segment_1.data,1)
    %Adding some dc offset to each channel to aid visualisation
    plot(time,interictal_segment_1.data(channel,:)+(channel-1)*0.3,'b','Linewidth',0.5),hold on;
    xlabel('Time{\it t} (ms)','Fontsize',13);
    ylabel('iEEG ({\muV})','Fontsize',13);
end
title('Interictal Segment 1 Raw Signals - Channels 1 - 16','Fontsize',16);
ylim([-0.2 4.7]);
yticks(([1:16]-1)*0.3);
yticklabels(interictal_segment_1.channels);

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000; %TO CHECK different values of the gain for the raw uV data
Voffset=1.5;
%Averaging across the 16 channels for each segment
Vin_ii(1,:)=mean(interictal_segment_1.data.*Gain+Voffset);
Vin_ii(2,:)=mean(interictal_segment_10.data.*Gain+Voffset);
Vin_ii(3,:)=mean(interictal_segment_20.data.*Gain+Voffset);
Vin_ii(4,:)=mean(interictal_segment_73.data.*Gain+Voffset);
Vin_ii(5,:)=mean(interictal_segment_159.data.*Gain+Voffset);
Vin_ii(6,:)=mean(interictal_segment_239.data.*Gain+Voffset);
Vin_ii(7,:)=mean(interictal_segment_358.data.*Gain+Voffset);
Vin_ii(8,:)=mean(interictal_segment_479.data.*Gain+Voffset);
Vin_ii(9,:)=mean(interictal_segment_498.data.*Gain+Voffset);
Vin_ii(10,:)=mean(interictal_segment_500.data.*Gain+Voffset);

seg_ii=[1,10,20,73,159,239,358,479,498,500];

figure(2);
%Obtaining the histograms of the linearly transformed data
for segment=1:10
    subplot(2,5,segment);
    histogram(Vin_ii(segment,:),50),hold on;
    
    %Mean and standard deviation of the specific segment data
    mean_segment_ii=mean(Vin_ii(segment,:));
    std_segment_ii=std(Vin_ii(segment,:));
    xline(mean_segment_ii,'Color','k','LineWidth',2);
    xline(mean_segment_ii-std_segment_ii,'Color','r','LineWidth',2,'LineStyle','--');
    xline(mean_segment_ii+std_segment_ii,'Color','r','LineWidth',2,'LineStyle','--');
    title({['Histogram of 16-channel averaged'] ['Interictal Segment ',num2str(seg_ii(segment))]},'Fontsize',14);
    xlabel('Voltage{\it V} (V)','Fontsize',13);
    ylabel('Counts','Fontsize',13);
    legend('Histogram bins',['Mean = ',num2str(mean_segment_ii),'V'],['Standard deviation = ',num2str(std_segment_ii*1000),'mV']);
end

%Averaging across the 10 segments
figure(3),subplot(2,1,1);
ensemble_ii=mean(Vin_ii);
histogram(ensemble_ii,50),hold on;
title(['Histogram for the ensemble of 10 interictal segments'],'Fontsize',17);
xlabel('Voltage{\it V} (V)','Fontsize',13);
ylabel('Counts','Fontsize',13);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_ii=mean(ensemble_ii);
std_ensemble_ii=std(ensemble_ii);
xline(mean_ensemble_ii,'Color','k','LineWidth',2);
xline(mean_ensemble_ii-std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_ii+std_ensemble_ii,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_ii),'V'],['Standard deviation = ',num2str(std_ensemble_ii*1000),'mV'],'Fontsize',13);

%Loading the preictal data (10 random segments)
load 'Dog_2_preictal_segment_0001.mat';
load 'Dog_2_preictal_segment_0005.mat';
load 'Dog_2_preictal_segment_0010.mat';
load 'Dog_2_preictal_segment_0015.mat';
load 'Dog_2_preictal_segment_0020.mat';
load 'Dog_2_preictal_segment_0025.mat';
load 'Dog_2_preictal_segment_0030.mat';
load 'Dog_2_preictal_segment_0035.mat';
load 'Dog_2_preictal_segment_0040.mat';
load 'Dog_2_preictal_segment_0042.mat';

%Data comes from iEEG which should be in μV, therefore dividing by 10^6
preictal_segment_1.data=preictal_segment_1.data.*unit; %sequence 
preictal_segment_5.data=preictal_segment_5.data.*unit; %sequence 
preictal_segment_10.data=preictal_segment_10.data.*unit; %sequence 
preictal_segment_15.data=preictal_segment_15.data.*unit; %sequence 
preictal_segment_20.data=preictal_segment_20.data.*unit; %sequence 
preictal_segment_25.data=preictal_segment_25.data.*unit; %sequence 
preictal_segment_30.data=preictal_segment_30.data.*unit; %sequence 
preictal_segment_35.data=preictal_segment_35.data.*unit; %sequence
preictal_segment_40.data=preictal_segment_40.data.*unit; %sequence 
preictal_segment_42.data=preictal_segment_42.data.*unit; %sequence 

%Obtaining the time axis given the specific sampling frequency
fs=round(preictal_segment_1.sampling_frequency);
time=[0:1/fs:(length(interictal_segment_500.data)-1)/fs];
%Plotting the raw signal for segment 1 to visualise
figure(1),subplot(2,1,2);
for channel=1:size(preictal_segment_1.data,1)
    %Adding some dc offset to each channel to aid visualisation
    plot(time,preictal_segment_1.data(channel,:)+(channel-1)*0.3,'r','Linewidth',0.5),hold on;
    xlabel('Time{\it t} (ms)','Fontsize',13);
    ylabel('iEEG ({\muV})','Fontsize',13);
end
title('Preictal Segment 1 Raw Signals - Channels 1 - 16','Fontsize',16);
ylim([-0.2 4.7]);
yticks(([1:16]-1)*0.3);
yticklabels(preictal_segment_1.channels);

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
%Averaging across the 16 channels
Vin_pi(1,:)=mean(preictal_segment_1.data*Gain+Voffset);
Vin_pi(2,:)=mean(preictal_segment_5.data*Gain+Voffset);
Vin_pi(3,:)=mean(preictal_segment_10.data*Gain+Voffset);
Vin_pi(4,:)=mean(preictal_segment_15.data*Gain+Voffset);
Vin_pi(5,:)=mean(preictal_segment_20.data*Gain+Voffset);
Vin_pi(6,:)=mean(preictal_segment_25.data*Gain+Voffset);
Vin_pi(7,:)=mean(preictal_segment_30.data*Gain+Voffset);
Vin_pi(8,:)=mean(preictal_segment_35.data*Gain+Voffset);
Vin_pi(9,:)=mean(preictal_segment_40.data*Gain+Voffset);
Vin_pi(10,:)=mean(preictal_segment_42.data*Gain+Voffset);

seg_pi=[1,5,10,15,20,25,30,35,40,42];
figure(4);
%Obtaining the histograms of the linearly transformed data
for segment=1:10
    subplot(2,5,segment);
    histogram(Vin_pi(segment,:),50);
    
    %Mean and standard deviation of the specific segment data
    mean_segment_pi=mean(Vin_pi(segment,:));
    std_segment_pi=std(Vin_pi(segment,:));
    xline(mean_segment_pi,'Color','k','LineWidth',2);
    xline(mean_segment_pi-std_segment_pi,'Color','r','LineWidth',2,'LineStyle','--');
    xline(mean_segment_pi+std_segment_pi,'Color','r','LineWidth',2,'LineStyle','--');
    title({['Histogram of 16-channel averaged'] ['Preictal Segment ',num2str(seg_pi(segment))]},'Fontsize',14);
    xlabel('Voltage{\it V} (V)','Fontsize',13);
    ylabel('Counts','Fontsize',13);
    legend('Histogram bins',['Mean = ',num2str(mean_segment_pi),'V'],['Standard deviation = ',num2str(std_segment_pi*1000),'mV']);
end

%Averaging across the 10 segments
figure(3),subplot(2,1,2);
ensemble_pi=mean(Vin_pi);
h_pi=histogram(ensemble_pi,50);
h_pi.FaceColor='r';
title(['Histogram for the ensemble of 10 preictal segments'],'Fontsize',17);
xlabel('Voltage{\it V} (V)','Fontsize',13);
ylabel('Counts','Fontsize',13);
%Getting the mean and standard deviation for the ensemble data
mean_ensemble_pi=mean(ensemble_pi);
std_ensemble_pi=std(ensemble_pi);
xline(mean_ensemble_pi,'Color','k','LineWidth',2);
xline(mean_ensemble_pi-std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
xline(mean_ensemble_pi+std_ensemble_pi,'Color','r','LineWidth',2,'LineStyle','--');
legend('Histogram bins',['Mean = ',num2str(mean_ensemble_pi),'V'],['Standard deviation = ',num2str(std_ensemble_pi*1000),'mV'],'Fontsize',13);

toc
%% Implementation in Memristor Array
% Test application of one-channel signal
% Takes about 9 minutes
clear,clc,close all;
tic

%Loading the interictal data (10 random segments)
load 'Dog_2_interictal_segment_0001.mat';

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000;
Voffset=1.5;

unit=10^(-6);

interictal_segment_1.data(1,:)=interictal_segment_1.data(1,:).*unit;
interictal_segment_1.data(1,:)=interictal_segment_1.data(1,:)*Gain+Voffset;

%Part 2: Signal Segmentation
%Trying for channel 1 of segment 1 first
channel_signal=[interictal_segment_1.data(1,:) zeros(1,42)]; %adding zero padding to segment to 16 memristors

%In this simplified case, the i-th memristor processes the i-th column
no_segments=64;
input_size=length(channel_signal)/no_segments;
channel_signal=reshape(channel_signal,[input_size,no_segments]);

%Defining the initial condition
Xic=1;
Roff=1500;
Ron=500;
Rinit=500;
Ginit=Xic/Ron+(1-Xic)/Roff;

%Defining the time vector
fs=interictal_segment_1.sampling_frequency;
time_vect=[0:1/fs:(length(channel_signal(:,1))-1)/fs];

%Pre-allocating for speed
dG1=zeros(1,no_segments);
dG2=zeros(1,no_segments);

%Iterating for each of the 64 columns/memristors
for memristor=1:no_segments
    %In this simplified case, the i-th memristor processes the i-th column
    inputsignal=channel_signal(:,memristor);
    inputsignal=inputsignal';
    
    %Calling the fODE2 function for the chalcogenide-based model
    [X,G,I,M]=chalcogenidemodel(inputsignal,time_vect,Xic);
    
    figure(memristor);
    %Variation of state variable X(t)
    subplot(1,5,1);
    plot(time_vect,X);
    title(['X(t) - memristor ',num2str(memristor)],'Fontsize',14);
    xlabel('Time{\it t} (s)','Fontsize',13);
    ylabel('{\itX}(t)','Fontsize',13);
    
    %Variation of memductance G(t)
    subplot(1,5,2);
    plot(time_vect,G);
    title(['G(t) - memristor ',num2str(memristor)],'Fontsize',14);
    xlabel('Time{\it t} (s)','Fontsize',13);
    ylabel('{\itG} (S)','Fontsize',13);
    
    %Variation of output current I(t)
    subplot(1,5,3);
    plot(time_vect,I);
    title(['I(t) - memristor (1, ',num2str(memristor),')'],'Fontsize',14);
    xlabel('Time{\it t} (s)','Fontsize',13);
    ylabel('{\itI} (A)','Fontsize',13);
    
    %Variation of memristance M(t)
    subplot(1,5,4);
    plot(time_vect,M);
    title(['M(t) - memristor ',num2str(memristor)],'Fontsize',14);
    xlabel('Time{\it t} (s)','Fontsize',13);
    ylabel('{\itM} (Ω)','Fontsize',13);
    
    %I-V characteristic
    subplot(1,5,5);
    plot(inputsignal,I);
    title(['I-V memristor ',num2str(memristor)],'Fontsize',14);
    xlabel('{\itV} (V)','Fontsize',13);
    ylabel('{\itI} (A)','Fontsize',13);
    
    disp(['Segment ',num2str(memristor),' processed in memristor ',num2str(memristor)]);
    
    %Way 1 of computing the dG
    dG1(memristor)=mean(G)-Ginit;
    
    %Way 2 of computing the dG
    deltaG=zeros(1,length(G)-1);
    for i=1:length(G)-1
        deltaG(i)=G(i+1)-G(i);
    end
    dG2(memristor)=mean(deltaG);
end
toc
%% Implementation in Memristor Array
% Test application of 16-channel interictal signal
% Takes about 3-4h per interictal segment
clear,clc,close all;
tic

%Loading each of the segments and run individually
load 'Dog_2_interictal_segment_0001.mat';

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000;
Voffset=1.5;

unit=10^(-6);

no_segments=64;

%Defining the initial condition
Xic=1;
Roff=1500;
Ron=500;
Rinit=500;
Ginit=Xic/Ron+(1-Xic)/Roff;

%Sampling frequency of signal
fs=interictal_segment_1.sampling_frequency;

for channel=1:length(interictal_segment_1.data(:,1))
    interictal_segment_1.data(channel,:)=interictal_segment_1.data(channel,:).*unit;
    interictal_segment_1.data(channel,:)=interictal_segment_1.data(channel,:).*Gain+Voffset;

    %Part 2: Signal Segmentation
    %Adding padding to segment to 64 memristors for each channel
    padding=zeros(1,42);
    for i=1:length(padding)
        padding(i)=interictal_segment_1.data(channel,end);
    end
    channel_signal=[interictal_segment_1.data(channel,:) padding];

    %The i-th memristor processes the i-th column
    input_size=length(channel_signal)/no_segments;
    channel_signal=reshape(channel_signal,[input_size,no_segments]);

    %Defining the time vector
    time_vect=[0:1/fs:(length(channel_signal(:,1))-1)/fs];

    %Iterating for each of the 64 memristors in every channel
    for memristor=1:no_segments
        %In this simplified case, the i-th memristor processes the i-th column
        inputsignal=channel_signal(:,memristor);
        inputsignal=inputsignal';

        %Calling the fODE2 function for the chalcogenide-based model
        [X,G,I,M]=chalcogenidemodel(inputsignal,time_vect,Xic);

        disp(['Segment ',num2str(memristor),' of channel ',num2str(channel),' processed in memristor (',num2str(channel),',',num2str(memristor),')']);

        %Way 1 of computing the dG
        dG1(channel,memristor)=mean(G)-Ginit;
        
        %Way 2 of computing the dG
        deltaG=zeros(1,length(G)-1);
        for i=1:length(G)-1
            deltaG(i)=G(i+1)-G(i);
        end
        dG2(channel,memristor)=mean(deltaG);
    end
end
toc

%% Implementation in Memristor Array
% Test application of 16-channel preictal signal
% Takes about 3-4h per preictal segment
clear,clc,close all;
tic

%Loading the preictal data (10 random segments)
load 'Dog_2_preictal_segment_0001.mat';

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000;
Voffset=1.5;

unit=10^(-6);

no_segments=64;

%Defining the initial condition
Xic=1;
Roff=1500;
Ron=500;
Rinit=500;
Ginit=Xic/Ron+(1-Xic)/Roff;

%Sampling frequency of signal
fs=preictal_segment_1.sampling_frequency;

for channel=1:length(preictal_segment_1.data(:,1))
    preictal_segment_1.data(channel,:)=preictal_segment_1.data(channel,:).*unit;
    preictal_segment_1.data(channel,:)=preictal_segment_1.data(channel,:)*Gain+Voffset;

    %Part 2: Signal Segmentation
    %Adding padding to segment to 64 memristors for each channel
    padding=zeros(1,42);
    for i=1:length(padding)
        padding(i)=preictal_segment_1.data(channel,end);
    end
    channel_signal=[preictal_segment_1.data(channel,:) padding];

    %The i-th memristor processes the i-th column
    input_size=length(channel_signal)/no_segments;
    channel_signal=reshape(channel_signal,[input_size,no_segments]);

    %Defining the time vector
    time_vect=[0:1/fs:(length(channel_signal(:,1))-1)/fs];

    %Iterating for each of the 64 memristors in every channel
    for memristor=1:no_segments
        %In this simplified case, the i-th memristor processes the i-th column
        inputsignal=channel_signal(:,memristor);
        inputsignal=inputsignal';

        %Calling the fODE2 function for the chalcogenide-based model
        [X,G,I,M]=chalcogenidemodel(inputsignal,time_vect,Xic);

        disp(['Segment ',num2str(memristor),' of channel ',num2str(channel),' processed in memristor (',num2str(channel),',',num2str(memristor),')']);

        %Way 1 of computing the dG
        dG1(channel,memristor)=mean(G)-Ginit;
        
        %Way 2 of computing the dG
        deltaG=zeros(1,length(G)-1);
        for i=1:length(G)-1
            deltaG(i)=G(i+1)-G(i);
        end
        dG2(channel,memristor)=mean(deltaG);
    end
end
toc

%% Memristor function block
function [X,G,I,M]=chalcogenidemodel(V,time_vect,Xic)

    %Solving the ODE
    [t X]=ode45(@(t,X) fODE(t,X,V,time_vect),time_vect,Xic);
    
    %Obtaining the conductance G as a function of X(t)
    Roff=1500;
    Ron=500;
    Rinit=500;
    G=X/Ron+(1-X)/Roff;
    
    %Obtaining the current I as a function of time t
    I=G.*V';
    
    %Obtaining the variation of the memristance with time
    M=V./(I');
    
    %Nested function which evaluates the ODE at each time instant
    function dXdt=fODE(t,X,V,time_vect)
        %Defining the constants
        beta_var=1/0.026; %Inverse of the thermal voltage VT
        Von=0.27;
        Voff=0.27;
        tau=.0001;

        V_interp=interp1(time_vect,V,t); %Interpolating the data set (Vt,V) at times t

        %Evaluating the ODE at times t
        dXdt=1/tau*[1./(1+exp(-beta_var.*(V_interp-Von))).*(1-X)-(1-1./(1+exp(-beta_var.*(V_interp+Voff)))).*X];
    end
end
