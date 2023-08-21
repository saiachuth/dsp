i)LTI System
%Without initializing values %y(n)+0.8y(n-2)+0.6y(n-3)=x(n)+0.7x(n-1)+0.5x(n-2)
 7
clear all;
close all;
b=input('Enter the coefficients of x '); a=input('Enter the coefficients of y '); N=input('Enter the length of the input sequence '); n=0:1:N;
step=1.^n;
imp=[1,zeros(1,N)];
RES1=filter(b,a,step) RES2=filter(b,a,imp)
subplot(2,2,1) stem(n,step)
grid on xlabel('Time'); ylabel('Amplitude'); title('Step Input ')
subplot(2,2,2) stem(n,imp)
grid on xlabel('Time'); ylabel('Amplitude'); title('Impulse Input')
subplot(2,2,3) stem(n,RES1)
grid on xlabel('Time'); ylabel('Amplitude'); title('Step Response')
subplot(2,2,4) stem(n,RES2)
grid on
xlabel('Input'); ylabel('Output Response'); title('Impulse Response')
