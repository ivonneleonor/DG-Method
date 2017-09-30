function [L2,h,uh] = DGsimplesolveMod23Nov(non,ss,penal)
%function  DGsimplesolve(nel,ss,penal) 
L=1;
% non  number of nodes  
nel=non-1; %number of elements 
h=(L-0)/(non-1); %  length between nodes  
Amat = (nel)*[0 0 0;0 4 0;0 0 16/3];
Bmat = (nel)*[penal 1-penal -2+penal; -ss-penal -1+ss+penal 2-ss-penal; 2*ss+penal 1-2*ss-penal -2+2*ss+penal] ;
Cmat = (nel)*[penal -1+penal -2+penal; ss+penal -1+ss+penal -2+ss+penal; 2*ss+penal -1+2*ss+penal -2+2*ss+penal];
Dmat = (nel)*[-penal -1+penal 2-penal; -ss-penal -1+ss+penal 2-ss-penal; -2*ss-penal -1+2*ss+penal 2-2*ss-penal] ;
Emat = (nel)*[-penal 1-penal 2-penal; ss+penal -1+ss+penal -2+ss+penal; -2*ss-penal 1-2*ss-penal 2-2*ss-penal];
F0mat =(nel)*[penal 2-penal -4+penal; -2*ss-penal -2+2*ss+penal 4-2*ss-penal; 4*ss+penal 2-4*ss-penal -4+4*ss+penal] ;
FNmat =(nel)*[penal -2+penal -4+penal; 2*ss+penal -2+2*ss+penal -4+2*ss+penal; 4*ss+penal -2+4*ss+penal -4+4*ss+penal];
% dimension of local matrices 
locdim = 3;
% dimension of global matrix
glodim = nel * locdim; 
%number of subintervals
n=glodim;
% initialize to zero matrix and right-hand side vector 
Aglobal = zeros(glodim,glodim); 
rhsglobal = zeros(glodim,1); 
% Gauss quadrature weights and points
wg(1) = 1.0; 
wg(2) = 1.0; 
sg(1) = -0.577350269189;
sg(2) = 0.577350269189;
% assemble global matrix and right-hand side 
% first block row 
for ii=1:locdim 
  for jj=1:locdim 
     %fprintf('\n');
     %fprintf('%d , %d',ii,jj);
     Aglobal(ii,jj) = Aglobal(ii,jj)+Amat(ii,jj)+F0mat(ii,jj)+Cmat(ii,jj);
     je = locdim+jj ;
     %fprintf('%d , %d',ii,je);
     Aglobal(ii,je) = Aglobal(ii,je)+Dmat(ii,jj);
     end; %jj 
   end; %ii

%  
%%for ii=1:locdim 
 %% for jj=1:locdim 
  %%   fprintf('%6.2f ',Aglobal(ii,jj)); 
   %% end; %jj 
  %% end; %ii
     
 
 % compute right-hand side
rhsglobal(1) = nel*penal;
rhsglobal(2) = nel*penal*(-1) - ss*2*nel;
rhsglobal(3) = nel*penal+ss*4*nel;

for ig=1:2 
   rhsglobal(1) = rhsglobal(1) + wg(ig)*sourcef((sg(ig)+1)/(2*nel))/(2*nel);  
   rhsglobal(2) = rhsglobal(2) + wg(ig)*sg(ig)*sourcef((sg(ig)+1)/(2*nel))/(2*nel);
   rhsglobal(3) = rhsglobal(3) + wg(ig)*sg(ig)*sg(ig)*sourcef((sg(ig)+1)/(2*nel))/(2*nel); 
   end; %ig
 rhsglobal(1);
 rhsglobal(2);
 rhsglobal(3);
   
   % intermediate block rows 
% loop over elements 
for i=2:(nel-1) 
  for ii=1:locdim 
      ie = ii+(i-1)*locdim; 
      for jj=1:locdim 
          fprintf('\n');        
          je = jj+(i-1)*locdim;
          %fprintf('%d , %d',ie,je);
          Aglobal(ie,je) = Aglobal(ie,je)+Amat(ii,jj)+Bmat(ii,jj)+Cmat(ii,jj);
          %fprintf(' %d, %f ',je,Aglobal);
          %Aglobal(ie,je)
          je = jj+(i-2)*locdim;
          Aglobal(ie,je) = Aglobal(ie,je)+Emat(ii,jj);
          je = jj+(i)*locdim;
          %fprintf('%d , %d',ie,je);
          Aglobal(ie,je)=Aglobal(ie,je)+Dmat(ii,jj);          
         % fprintf('%d , %d, %f ',ie,je,Aglobal(ie,je));
       end; %jj
      % compute right-hand side 
      for ig=1:2 
           rhsglobal(ie) = rhsglobal(ie)+wg(ig)*(sg(ig)^(ii-1))*sourcef((sg(ig)+2*(i-1)+1.0)/(2*nel))/(2*nel);  
        %  fprintf('\n');             
           %fprintf('ig= %d, sg(ig)= %f,i= %d nel=%d, 2*nel= %d, b=%f',ig,sg(ig),i,nel,2*nel, sourcef((sg(ig)+2*(i-1)+1.0)/(2*nel))/(2*nel));
         % fprintf(' %d, %d, %f, %f, %f, %f, %f ', ie, ii,  wg(ig), sg(ig) ,(sg(ig)^(ii-1)),sourcef((sg(ig)+2*(i-1)+1.0)/(2*nel))/(2*nel), rhsglobal(ie));
      end; %
   end; %ii 
 end; %i
%fprintf('s=%f',sourcef(1.0));
 
 % last block row
for ii=1:locdim
 ie = ii+(nel-1)*locdim;       
      for jj=1:locdim  
       je = jj+(nel-1)*locdim;
       Aglobal(ie,je) = Aglobal(ie,je)+Amat(ii,jj)+FNmat(ii,jj)+Bmat(ii,jj); 
       je = jj+(nel-2)*locdim; 
       Aglobal(ie,je) = Aglobal(ie,je)+Emat(ii,jj); 
   
      end; %jj
  
  % compute right-hand side 
  for ig=1:2
    
      rhsglobal(ie) = rhsglobal(ie)+wg(ig)*(sg(ig)^(ii-1))*sourcef((sg(ig)+2*(nel-1)+1.0)/(2*nel))/(2*nel); 
    
       end; %ig 
       end; %ii

   


% solve linear system
ysol = Aglobal\rhsglobal;
x=linspace(0,1,nel*3);
yanal=(1-x).*exp(-x.*x);
%plot(x,ysol,x,yanal)

j=1;

%fprintf('xm(32)=%f',xm(32));
%  for i=1:31
%        fprintf('\n');
%        fprintf(' %f \n',xm(i));
%   end;      
              

for i=1:nel
   ul(i)=ysol(j)-ysol(j+1)+ysol(j+2); 
   ur(i)=ysol(j)+ysol(j+1)+ysol(j+2);
   j=j+3;
   xl(i)=(i-1)/nel;
   xr(i)=(i)/nel;
 %  plot(xl,ul,xl,ur)
end;   


%L2,h,uh

%% create  uh  variable  
uh=zeros(1,non);

for i=1:non
   uh(i)=ysol(i);
end;
%disp('this is  ur(3)');
%ur(3)
%%  Here  we  put  the  error Function and  compute  the L2  norm 
% gauss points 
% uh  dg points 
% non  number nodes 
 L2= errorv2( 10,uh,non,0,1);
 
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yval = sourcef(xval)
% source function for exact solution=(1-x)e?(-x?2)
yval = -(2*xval-2*(1-2*xval)+4*xval*(xval-xval^2))*exp(-xval*xval);
return;

function pol1=P1(x)
pol1=1;
return;

function pol2=P2(x,nel,n)
h=1/nel;
pol2=(2/h)*(x-(n+0.5)*h);
return;


function pol3=P3(x,nel,n)
h=1/nel;
pol3=(4/(h*h))*((x-(n+0.5)*h)^2);
return;
