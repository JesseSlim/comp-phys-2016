clear all;
close all;
clc;

load data.mat

dens  = linspace(0.1, 2, size(cv,1))';
temp = linspace(0.1, 4, size(cv,2));

[Temp, Dens] = meshgrid(temp,dens);

figure();
hold on;
for i=1:1:20
    cv(i,:)
    plot(temp, cv(i,:));
    pause(1)
    axis([0 2 -3 5])
end

% surf(Dens, Temp, cv, 'EdgeColor', 'none');
% axis([0.5 2 0.1 4 0 4]);
% caxis([-12, 50]);
% for (i=1; i<20; i++)
% colorbar;
% view(90,90);
