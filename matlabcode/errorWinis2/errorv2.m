function [ l2norm ] = errorv2( nq,uh,nOn,a,b)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%%--------------------------------------------------------------------
%       THis  function computes  the L2 norm of  the  error
%       || u-uh ||_{ L2(0,L) } given  an analitical  solution u 
%       of  the POisson equation  in 1d  and  a mesh size h
%
%---------------------------------------------------------------------
% nOn  number of nodes
% u is  analitical solution  function 
% L length  of  the  domain.  
% nq   % number  of  quadrature  points necesary   to  obtain an excact  
       % solution  given  that (u-uh)^{2} it's a polinomioum of 4th degree.
       % nq=2(3)-1=5,  so  in our  case 3 its  enough.
% uh finite element  aproximation

nOe=nOn-1;    % numeberof  elements       
%%f=@(x) 0.5 * x * ( (b-a) - x ); % analitycal solution 
f=@(x) ( 1 - x )*exp(-x*x);

x=linspace(a,b,nOe+1); % this  is  the  1D domain divided in nOe elements
xs=0.0;  % x start  of element 
xe=0.0;  % x end of element
us=0.0;  % u start 
ue=0.0;  % u end 
J=0.0;   % Jacobian

    
Integrals=zeros(1,nOe); % We set a vector  for the integral of each element
iaux=1;
for i=1:nOe
 xs=x(i);
 xe=x(i+1);
 us=uh(iaux);
 um=uh(iaux+1);
 ue=uh(iaux+2);
 iaux=iaux+3;
 %um=0.5*(us+ue);
 J=(xe-xs)*0.5;
 %%xm=(xs+xe)*0.5; %% middle  point 
 sum=0.0;
 x_map=zeros(1,nq);
 f_map=zeros(1,nq);
 [eta,w]=lgwt(nq,-1,1);
 u_map=zeros(1,nq);
 N1=0.0;
 N2=0.0;
 error_map   = 0.0;

 for j=1:nq
      
      N1=( 1.0 + eta(j) )*0.5;
      N2=( 1.0 - eta(j) )*0.5;
      %N1=0.5*eta(j)*( eta(j)-1.0 );
      %N2=(1-eta(j))*(1+eta(j));
      %N3=0.5*eta(j)*( eta(j)+1.0 );
      
      x_map(j)=N2 * xs + N1 * xe; 
      u_map(j)= us +  um * eta(j) + ue * eta(j) * eta(j);
      
      %x_map(j)=N1 * xs + N2 * xm +N3 * xe;
      %u_map(j)=N1 * us + N2 * um +N3 * ue;
      
      error_map= ( f(x_map(j)) - u_map(j) )*( f(x_map(j))  - u_map(j) ); % (u-uh)^2
      %%J=( eta(j) - 0.5 ) * xs - 2 * eta(j) * xm + ( eta(j) + 0.5 ) * xe;
      sum= error_map * w ( j ) * J  + sum; 
      
 end
 Integrals(i)=sum;
    
    
end

I=0;
for k=1:nOe
    I=Integrals(k)+I;
end

l2norm=sqrt(I);
