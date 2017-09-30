%------------------------
%   This  program  computes  the error =||u-uh||L2
%   against the  mesh  size  h = 1/2^{i} for i=2,3...numpts  
%   where  numpts  is the  number  of  points  that  we  want  for  our
%   plot. 
%   
%------------------------
clear all 
close all
clc
ss=-1;
penal=2;
numpts=10;   %  numer  of  points  to  plot 

hv=zeros(1,numpts);
L2normv=zeros(1,numpts);
%size(hv)
for i=2:numpts
 %[L2norm,h]=poisson1dv1(2^(i));
 [L2norm,h] = DGsimplesolveMod23Nov(2^(i)+1,ss,penal);
 hv(i-1)=h;            %h  vector
 L2normv(i-1)=L2norm; 

 

end
 
for i=2:numpts-1
s(i)=log(L2normv(i)/L2normv(i+1))/log(hv(i)/hv(i+1));
end

s
%hv=fliplr(hv)

L2normv
hv
%size(hv)?)

loglog(hv, L2normv,'r*-')
title('L2 norm vs h ')
ylabel('L2 norm ')
xlabel('h')