%% Final Year Project
% Implementation in HP Memristor Array
% Application of 16-channel interictal signal
clear,clc,close all;
tic

%Loading each interictal segment and running the section
load 'Dog_2_interictal_segment_0500.mat';

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000;
Voffset=1.5;

unit=10^(-6);

no_segments=64;

%Defining the initial condition and device parameters
Xic=1;
Ron=100;
Roff=30000;
mu=10^(-10)*10^(-4);
D=7*10^(-9); %To investigate variable thickness - either 7nm or 8nm
Ginit=Xic/Ron+(1-Xic)/Roff;

%Sampling frequency of signal
fs=interictal_segment_500.sampling_frequency;

for channel=1:16
    interictal_segment_500.data(channel,:)=interictal_segment_500.data(channel,:).*unit;
    interictal_segment_500.data(channel,:)=interictal_segment_500.data(channel,:)*Gain+Voffset;

    %Part 2: Signal Segmentation
    %Adding padding to segment to 64 memristors for each channel
    padding=zeros(1,42);
    for i=1:length(padding)
        padding(i)=interictal_segment_500.data(channel,end);
    end
    channel_signal=[interictal_segment_500.data(channel,:) padding];

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

        %Calling the HP memristor function block 
        [I,G]=hpmodel(Ron,Roff,mu,D,inputsignal,time_vect,fs);

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
%% Implementation in HP Memristor Array
% Test application of 16-channel preictal signal
clear,clc,close all;
tic

%Loading each preictal segment and running the section
load 'Dog_2_preictal_segment_0035.mat';

%Part 1: Linear transformation of raw signals to bring to desired range (1-2V)
Gain=1000000;
Voffset=1.5;

unit=10^(-6);

no_segments=64;

%Defining the initial condition
Xic=1;
Ron=100;
Roff=30000;
mu=10^(-10)*10^(-4);
D=7*10^(-9); %To investigate variable thickness - either 7nm or 8nm
Ginit=Xic/Ron+(1-Xic)/Roff;

%Sampling frequency of signal
fs=preictal_segment_35.sampling_frequency;

for channel=1:length(preictal_segment_35.data(:,1))
    preictal_segment_35.data(channel,:)=preictal_segment_35.data(channel,:).*unit;
    preictal_segment_35.data(channel,:)=preictal_segment_35.data(channel,:)*Gain+Voffset;

    %Part 2: Signal Segmentation
    %Adding padding to segment to 64 memristors for each channel
    padding=zeros(1,42);
    for i=1:length(padding)
        padding(i)=preictal_segment_35.data(channel,end);
    end
    channel_signal=[preictal_segment_35.data(channel,:) padding];
    
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

        %Calling the HP memristor function block 
        [I,G]=hpmodel(Ron,Roff,mu,D,inputsignal,time_vect,fs);

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

%% HP memristor model function
function [I,G]=hpmodel(Ron,Roff,mu,D,V,time_vect,fs)
%INPUTS to MODEL:
%ON resistance
%OFF resistance
%μ: average vacancy mobility of linear ionic drift
%Thickness D of the memristor (with high and low dopant concentration)
%Voltage V at times time_vect

%OUTPUTS of MODEL:
%Current I at times t
%Memristance M at times t

    %Defining the constants k2 and k3
    k2=(Ron-Roff)*mu*Ron/(D^2);
    k3=Roff/10;
    
    I=zeros(1,length(time_vect));
    G=zeros(1,length(time_vect));
    
    for t=1:length(time_vect)
       
        %Obtaining the current I as a function of time t
        %I(t)=V(t)/((k3-2*k2*integral(V_time,0,time_vect(t)))^(1/2));
        
        V_time = V(1:t);
        
        I(t)=V(t)/((k3-2*k2*1/fs*trapz(V_time))^(1/2));
    
        %Obtaining the variation of the memristance with time
        G(t)=1/(V(t)/(I(t))');
    end
end