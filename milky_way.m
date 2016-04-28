%   generate a random milky way galaxy
axis([-5 5 -5 5 -5 5]);
noise = randn(3,1000);
noise(2,:)=noise(2,:)./3;
scatter3(noise(1,:),noise(2,:),noise(3,:));
axis([-5 5 -5 5 -5 5]);