% Demo for EEG preprocess in BMEG3330
% 1/19/2023
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

% regress EOG
tmpX = [ones(length(X),1),X];
b1 = regress(Y1,tmpX);
b2 = regress(Y2,tmpX);
b3 = regress(Y3,tmpX);

Y1_hat = Y1 - b1(2).*X;
Y2_hat = Y2 - b2(2).*X;
Y3_hat = Y3 - b3(2).*X;

figure;
subplot(4,1,1);
plot(EEG.times,X,'r');title('VEOG');
legend('VEOG')

subplot(4,1,2);
plot(EEG.times,Y1,'b',EEG.times,Y1_hat,'r');title('FP1');
legend('raw','clean')

subplot(4,1,3);
plot(EEG.times,Y2,'b',EEG.times,Y2_hat,'r');title('FP2');
legend('raw','clean')

subplot(4,1,4)
plot(EEG.times,Y3,'b',EEG.times,Y3_hat,'r');title('Cz')
legend('raw','clean')



% t=EEG.times;
% wtY3 = modwt(Y3,'db8',5);
% figure;
% for i=1:6
% subplot(6,1,i);plot(t,wtY3(i,:));
% end

load Espiga3%https://doi.org/10.1007/11578079_96.
E = Espiga3(:,1);
t=0:1/200:994/200;

wt = modwt(E,'db8',5);
figure;
for i=1:6
subplot(6,1,i);plot(t,wt(i,:));
end


% emd(E,'MaxNumIMF',5)
