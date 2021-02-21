% code to test SFD_depth_calculator.m 

% define spatial frequencies of interest
fxs=[0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.5];
% define CDF levels for output
cdflevels=[10 25 50 75 90];

% trial 1: human skin 731 nm
mua=0.2437; % absorption coefficient mua, units[/mm]
musp=2.6814; % reduced scattering coefficient mus', units[/mm]
% SFD_depth_calculator returns a matrix of size [5,length(fxs)] 
% where the first index 5 = CDF levels [10 25 50 75 90]
human_skin_depths=SFD_depth_calculator(mua,musp,fxs);
% print out results
disp(sprintf('mus''/mua = %5.3f l*=%5.3f',musp/mua,1/(mua+musp)));
for i=1:5 % number of CDF levels
  disp('----------');
  disp(sprintf('CDF level = %d percent',cdflevels(i)));
  disp('fx[/mm]  depth[mm]');
  for j=1:length(fxs)
    disp(sprintf('%4.3f\t %5.4f',fxs(j),human_skin_depths(i,j)));
  end
end
% plot results
figure;
% plot CDF levels 10 and 90 
hold on;
errorbar(fxs,human_skin_depths(3,:),human_skin_depths(3,:)-human_skin_depths(1,:),human_skin_depths(5,:)-human_skin_depths(3,:),'r-','Linewidth',2);
% plot CDF levels  and 90 
hold on;
errorbar(fxs,human_skin_depths(3,:),human_skin_depths(3,:)-human_skin_depths(2,:),human_skin_depths(4,:)-human_skin_depths(3,:),'b-','Linewidth',2);
% plot mean depth [CDF level 50]
plot(fxs,human_skin_depths(3,:),'k-','LineWidth',2);
xlabel('fx [/mm]','FontSize',20);
ylabel('median depth [mm]','FontSize',20);
title('human skin \lambda=731nm','FontSize',20);
legend('[10-90]%','[25-75]%','50%');
set(gca,'TickDir','out','FontSize',20);
saveas(gca,'../results/human_skin_731nm_depths.png');




