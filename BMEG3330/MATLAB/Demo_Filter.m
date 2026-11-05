clear,clc;

eeglab;
close all;
EEG = pop_loadset('filename','DemoData123.set','filepath','/Users/wlou/matlab/BMEG3330_EEG/W02_Demo/');
% EEG = pop_loadset('filename','DemoData123_250Hz.set','filepath','/Users/wlou/Library/CloudStorage/OneDrive-TheChineseUniversityofHongKong/Teaching/2023-2024/BMEG3330/Slides/W03/');

InputData = double(transpose(EEG.data(15,:)));%Cz Channel
fs=EEG.srate;           % sampling frequency
fnq = fs/2;             %% Nyquist frequency

t = EEG.times;
fband = [8 12];     % alpha  band
Wn = fband/fnq;  
ordbf = 3;
[bfilt,afilt] = butter(ordbf, Wn);
% [bfilt,afilt] = butter(ordbf, Wn,'stop');
InputData = detrend(InputData);
OutputData = filtfilt(bfilt,afilt,InputData);
figure;
subplot(2,1,1);
plot(t,InputData,'b');
hold on;plot(t,OutputData,'r','lineWidth',1);
legend('Raw Data','Filtered Data')


L = length(OutputData);
Y = fft(OutputData);
P2 =abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*((0:L/2)/L);
subplot(2,1,2);
fIdx = f<30;
plot(f(fIdx),P1(fIdx),'r','lineWidth',1);
xlabel('Frequency (Hz)');ylabel('|P1(f)|')


rY = fft(InputData);
rP2 =abs(rY/L);
rP1 = rP2(1:L/2+1);
rP1(2:end-1) = 2*rP1(2:end-1);
hold on;plot(f(fIdx),rP1(fIdx),'b');