function [ysol,ul,ur,xl,xr]=DGsimplesolve3(nel,ss,penal)
%function [x0,fl0,rr0,it0,rv0,S,rhsglobal] = DGsimplesolve3(nel,ss,penal)
%function  DGsimplesolve(nel,ss,penal) 

format long

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
      %if((ie==12)&&(je==12))
      %      fprintf('ie=%d, je=%d, Aglobal=%f, ii=%d,jj=%d, Amat=%f, FNmat=%f, Bmat=%f',ie, je, Aglobal(ie,je),ii,jj,Amat(ii,jj),FNmat(ii,jj),Bmat(ii,jj));
      %end;
       Aglobal(ie,je) = Aglobal(ie,je)+Amat(ii,jj)+FNmat(ii,jj)+Bmat(ii,jj); 
       je = jj+(nel-2)*locdim; 
       Aglobal(ie,je) = Aglobal(ie,je)+Emat(ii,jj); 
      % fprintf('%f', Aglobal(ie,je));
      % fprintf('\n');
      end; %jj
  
  % compute right-hand side 
  for ig=1:2
     % fprintf('\n');
      rhsglobal(ie) = rhsglobal(ie)+wg(ig)*(sg(ig)^(ii-1))*sourcef((sg(ig)+2*(nel-1)+1.0)/(2*nel))/(2*nel); 
      %fprintf('ig=%d,  sg(ig)= %f, ii-1= %d c=%f, d=%f  ',ig,sg(ig),ii-1,sg(ig)^(ii-1),sourcef((sg(ig)+2*(nel-1)+1.0)/(2*nel))/(2*nel));
      % fprintf('%f \n',rhsglobal(ie));
       end; %ig 
       end; %ii

       
%    fprintf('Aglobal(11,11)=%f',Aglobal(12,12)); 
%   for i=1:glodim
%        fprintf('\n');
%        for j=1:glodim
%             fprintf(' %f ',Aglobal(i,j));
%        end;
%   end;

  % for i=1:glodim
  %      fprintf('\n');
  %      fprintf(' %f \n',rhsglobal(i));
  % end;

 
n1=length(rhsglobal);  
M1 = spdiags((1:n1)',0,n1,n1);


tol = 1e-10;
maxit = 100000;

%pcg(A,b1,tol,maxit,M1);
% solve linear system
S = sparse(Aglobal);
%ysol2 = cgs(Aglobal,rhsglobal,tol,maxit);
%ysol3 = cgs(S,rhsglobal,tol,maxit);

x1 = pcg(S,rhsglobal,tol,maxit);

%x1 = pcg(S,rhsglobal,tol,maxit,M1)

ysol = Aglobal\rhsglobal;

%[x0,fl0,rr0,it0,rv0] = pcg(Aglobal,rhsglobal,1e-8,100);


x=linspace(0,1,nel*3);
yanal=(1-x).*exp(-x.*x);

%plot(x,ysol,x,yanal)

%plot(x,ysol)


j=1;

%fprintf('xm(32)=%f',xm(32));
%  for i=1:31
%        fprintf('\n');
%        fprintf(' %f \n',xm(i));
%   end;      
 

%fileID = fopen('CudaOut.txt','r');

%A = fscanf(fileID,'%f');

%for i=1:nel
%   ul(i)=A(j)-A(j+1)+A(j+2); 
%   ur(i)=A(j)+A(j+1)+A(j+2);
%   j=j+3;
%   xl(i)=(i-1)/nel;
 %  xr(i)=(i)/nel;
 %  plot(xl,ul,xl,ur)
%end; 



for i=1:nel
   ul(i)=ysol(j)-ysol(j+1)+ysol(j+2); 
   ur(i)=ysol(j)+ysol(j+1)+ysol(j+2);
   j=j+3;
   xl(i)=(i-1)/nel;
   xr(i)=(i)/nel;
 %  plot(xl,ul,xl,ur)
end;   



plot(xl(1),ul(1),'bo',xr(1),ur(1),'r*')
hold on
plot(xl(2),ul(2),'bo',xr(2),ur(2),'r*')
hold on
plot(xl(3),ul(3),'bo',xr(3),ur(3),'r*')
hold on
plot(xl(4),ul(4),'bo',xr(4),ur(4),'r*')
hold on
plot(xl(5),ul(5),'bo',xr(5),ur(5),'r*')
hold on
plot(xl(6),ul(6),'bo',xr(6),ur(6),'r*')
hold on
plot(xl(7),ul(7),'bo',xr(7),ur(7),'r*')
hold on
plot(xl(8),ul(8),'bo',xr(8),ur(8),'r*')
hold on
plot(xl(9),ul(9),'bo',xr(9),ur(9),'r*')
hold on
plot(xl(10),ul(10),'bo',xr(10),ur(10),'r*')
hold on
%plot(x,yanal)
%hold on

xx(1)=xl(1);
xx(2)=xr(1);
xx(3)=xl(2);
xx(4)=xr(2);
xx(5)=xl(3);
xx(6)=xr(3);
xx(7)=xl(4);
xx(8)=xr(4);
xx(9)=xl(5);
xx(10)=xr(5);
xx(11)=xl(6);
xx(12)=xr(6);
xx(13)=xl(7);
xx(14)=xr(7);
xx(15)=xl(8);
xx(16)=xr(8);
xx(17)=xl(9);
xx(18)=xr(9);
xx(19)=xl(10);
xx(20)=xr(10);

ll(1)=ul(1);
ll(2)=ur(1);
ll(3)=ul(2);
ll(4)=ur(2);
ll(5)=ul(3);
ll(6)=ur(3);
ll(7)=ul(4);
ll(8)=ur(4);
ll(9)=ul(5);
ll(10)=ur(5);
ll(11)=ul(6);
ll(12)=ur(6);
ll(13)=ul(7);
ll(14)=ur(7);
ll(15)=ul(8);
ll(16)=ur(8);
ll(17)=ul(9);
ll(18)=ur(9);
ll(19)=ul(10);
ll(20)=ur(10);

%plot(x,yanal,'b')
%hold on
%plot(xx,ll,'r*')
%hold on


%xr=linspace(0,1,nel);
%plot(x,yanal,'b',xr,ur,'r',xl,ul,'g','linewidth',2)
%plot(x,ysol,'r',x,yanal,'b')
%plot(xr,ur)


%j=1;

 %for i=1:nel
 %   PC(j)=P1(x(i));
 %   PC(j+1)=P2(x(i),nel,n);
 %   PC(j+2)=P3(x(i),nel,n);
 %   j=j+3;
 %      end; 
 
%j=1;

 


%xl=linspace(0,1,nel);
%plot(xl,ul,'r',x,yanal,'b')
%plot(x,ysol,'r',x,yanal,'b')


%k=3;
%j=1;
%for i=1:nel-1
%   ur(i)=ysol(k)*PC(j)+ysol(k+1)*PC(j+1)+ysol(k+2)*PC(j+2);
%   j=j+3;
%   k=k+3;
%end; 

%xr=linspace(0,1,nel-1);
%plot(xr,ur,'r',x,yanal,'b')

%for i=1:nel-1
%   um(i)=(ul(i)-ur(i))/2;
%       end; 
%plot(xr,um,'r',x,yanal,'b')
%plot(xr,um)
 %plot(xm,u)
  
 
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
%fprintf('h=%f\n',h);
%fprintf('4/(h*h)=%f\n',4/(h*h));
%fprintf('((x-(n+0.5)*h)^2)=%f\n',((x-(n+0.5)*h)^2));
pol3=(4/(h*h))*((x-(n+0.5)*h)^2);
return;








       