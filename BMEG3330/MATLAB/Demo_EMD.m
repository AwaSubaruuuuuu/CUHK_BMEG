eeglab;
close all;
EEG = pop_loadset('/Users/wlou/matlab/BMEG3330_EEG/W02_Demo/DemoData123.set');

X = transpose(EEG.data(32,:));%VEOG channel
Y1 = transpose(EEG.data(30,:));% FP1 Channel
Y2 = transpose(EEG.data(28,:));% FP2 Channel
Y3 = transpose(EEG.data(15,:));%Cz Channel

% detrend
X = detrend(X,1);
Y1 = detrend(Y1,1);
Y2 = detrend(Y2,1);
Y3 = detrend(Y3,1);

t = EEG.times;

% wtY1 = modwt(Y1,'db8',5);
% figure;
% subplot(8,1,1)
% plot(t,X,'r');title('VEOG')
% 
% subplot(8,1,2)
% plot(t,Y1,'b');title('FP1')
% for i=1:6
% subplot(8,1,i+2);plot(t,wtY1(i,:));
% end
% 
% 
% figure;
% wtY3 = modwt(Y3,'db8',5);
% subplot(8,1,1)
% plot(t,X,'r');title('VEOG')
% 
% subplot(8,1,2)
% plot(t,Y3,'b');title('Cz')
% for i=1:6
% subplot(8,1,i+2);plot(t,wtY3(i,:));
% end


emd(Y1,'MaxNumIMF',5)