A = delsq(numgrid('S',10));
b = ones(size(A,1),1);
[x0,fl0,rr0,it0,rv0] = pcg(A,b,1e-8,100);