%   generate a random milky way galaxy
%axis([-5 5 -5 5 -5 5]);
%noise = randn(3,1000);
%noise(2,:)=noise(2,:)./3;
%scatter3(noise(1,:),noise(2,:),noise(3,:));
%axis([-5 5 -5 5 -5 5]);

clear

uk_food = [105 103 103 66; 245 227 242 267; 685 803 750 586;
147 160 122 93; 193 235 184 209; 156 175 147 139;
720 874 566 1033; 253 265 171 143; 488 570 418 355;
198 203 220 187; 360 365 337 334; 1102 1137 957 674;
1472 1582 1462 1494; 57 73 53 47; 1374 1256 1572 1506;
375 475 458 135; 54 64 62 41];

%   demonstrate PCA with UK food consumption
c = cellstr({'Eng' 'Scot' 'Wal' 'N Ire'});

coefs = pca(uk_food');%rows must be observations
coefs2 = princomp(uk_food');

pca1 = coefs2(:,1);
twocols = coefs2(:,1:2);

num = pca1'*uk_food;
scatter(num,[0 0 0 0])
text(num,[0 0 0 0],c);

%num = twocols'*uk_food;
%scatter(num(1,:),num(2,:))
%text(num(1,:),num(2,:),c);