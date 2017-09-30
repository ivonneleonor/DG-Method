#include <math.h>
#include "stdio.h" 


  double sourcef(double xval)
    {
        //source function for exact solution = (1-x)e^(-x^2)
         double yval;
         yval=-(2*xval-2*(1-2*xval)+4*xval*(xval-pow(xval,2)))*exp(-xval*xval);

        return yval;
    }


int main(int argc, char **argv)

 {
//number of subintervals
    const int nel=4,mz=3;

    int glodim,je,ie;
    double ss,penal;
    double Amat[3][3],Bmat[3][3],Cmat[3][3],Dmat[3][3],Emat[3][3],F0mat[3][3],FNmat[3][3];

FILE *f = fopen("Matrix.txt", "w");
FILE *g = fopen("rhs.txt", "w");
if (f == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}
if (g == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}
//symmetric interior penaltyGalerking (SIPG) method   
//    ss=-1;
//    penal=-1;

//global element method
//    ss=0;
//    penal=-1;

//nonsymmetric interior penalty Galerkin (NIPG) method
    ss=1.0;
    penal=1.0;

//NIPG 0 method
//    ss=0;
//    penal=1;

//dimension of global matrix
glodim = nel * mz;

double Aglobal[glodim][glodim];
double rhsglobal[glodim];

    for(int j=0;j<glodim;j++)
    {  
       for(int i=0;i<glodim;i++)
        {
           Aglobal[i][j]=0.0;
        }
    }

    for(int i=0;i<glodim;i++)
        {
           rhsglobal[i]=0.0;
        }


    Amat[0][0]=0.0;
    Amat[0][1]=0.0;
    Amat[0][2]=0.0;
    Amat[1][0]=0.0;
    Amat[1][1]=4.0;
    Amat[1][2]=0.0;
    Amat[2][0]=0.0;
    Amat[2][1]=0.0;
    Amat[2][2]=(16.0/3.0);
    
    for(int j=0;j<mz;j++)
    {  
      //  printf("\n");
        for(int i=0;i<mz;i++)
        {
            Amat[i][j]=nel*Amat[i][j];
     //       printf(" %f ",Amat[i][j]);
        }
    }
    
   //  printf("\n");

    Bmat[0][0]=penal;
    Bmat[0][1]=1.0-penal;
    Bmat[0][2]=-2.0+penal;
    Bmat[1][0]=-ss-penal;
    Bmat[1][1]=-1.0+ss+penal;
    Bmat[1][2]=2.0-ss-penal;
    Bmat[2][0]=2.0*ss+penal;
    Bmat[2][1]=1.0-2.0*ss-penal;
    Bmat[2][2]=-2.0+2.0*ss-penal;


    for(int i=0;i<mz;i++)
    {  
     //   printf("\n");
        for(int j=0;j<mz;j++)
        {
            Bmat[i][j]=nel*Bmat[i][j];
      //      printf(" %f ",Bmat[i][j]);
        }
    }

    // printf("\n");

    Cmat[0][0]=penal;
    Cmat[0][1]=-1+penal;
    Cmat[0][2]=-2+penal;
    Cmat[1][0]=ss+penal;
    Cmat[1][1]=-1+ss+penal;
    Cmat[1][2]=-2+ss+penal;
    Cmat[2][0]=2*ss+penal;
    Cmat[2][1]=-1+2*ss+penal;
    Cmat[2][2]=-2+2*ss+penal;


    for(int i=0;i<mz;i++)
    {  
      //  printf("\n");
        for(int j=0;j<mz;j++)
        {
            Cmat[i][j]=nel*Cmat[i][j];
     //       printf(" %f ",Cmat[i][j]);
 
        }
    }

   // printf("\n");

    Dmat[0][0]=-penal;
    Dmat[0][1]=-1+penal;
    Dmat[0][2]=2-penal;
    Dmat[1][0]=-ss-penal;
    Dmat[1][1]=-1+ss+penal;
    Dmat[1][2]=2-ss-penal;
    Dmat[2][0]=-2*ss-penal;
    Dmat[2][1]=-1+2*ss+penal;
    Dmat[2][2]=2-2*ss-penal;

    for(int i=0;i<mz;i++)
    {  
//	printf("\n");
        for(int j=0;j<mz;j++)
        {
            Dmat[i][j]=nel*Dmat[i][j];
  //          printf(" %f ",Dmat[i][j]);    
        }
    }

  //  printf("\n");

    Emat[0][0]=-penal;
    Emat[0][1]=1-penal;
    Emat[0][2]=2-penal;
    Emat[1][0]=ss+penal;
    Emat[1][1]=-1+ss+penal;
    Emat[1][2]=-2+ss+penal;
    Emat[2][0]=-2*ss-penal;
    Emat[2][1]=1-2*ss-penal;
    Emat[2][2]=2-2*ss-penal;

    for(int i=0;i<mz;i++)
    {  
      //  printf("\n");
        for(int j=0;j<mz;j++)
        {
            Emat[i][j]=nel*Emat[i][j];
        //    printf(" %f ",Emat[i][j]);
        }
    }

   // printf("\n");

    F0mat[0][0]=penal;
    F0mat[0][1]=2-penal;
    F0mat[0][2]=-4+penal;
    F0mat[1][0]=-2*ss-penal;
    F0mat[1][1]=-2+2*ss+penal;
    F0mat[1][2]=4-2*ss-penal;
    F0mat[2][0]=4*ss+penal;
    F0mat[2][1]=2-4*ss-penal;
    F0mat[2][2]=-4+4*ss+penal;
  
    for(int i=0;i<mz;i++)
    {  
      //  printf("\n");
        for(int j=0;j<mz;j++)
        {
            F0mat[i][j]=nel*F0mat[i][j];
        //    printf(" %f ",F0mat[i][j]);
        }
    }

    // printf("\n");

    FNmat[0][0]=penal;
    FNmat[0][1]=-2+penal;
    FNmat[0][2]=-4+penal;
    FNmat[1][0]=2*ss+penal;
    FNmat[1][1]=-2+2*ss+penal;
    FNmat[1][2]=-4+2*ss+penal;
    FNmat[2][0]=4*ss+penal;
    FNmat[2][1]=-2+4*ss+penal;
    FNmat[2][2]=-4+4*ss+penal;

    for(int i=0;i<mz;i++)
    {  
       // printf("\n");  
        for(int j=0;j<mz;j++)
        {
            FNmat[i][j]=nel*FNmat[i][j];
          //  printf(" %f ",FNmat[i][j]);
        }
    }

   // printf("\n");
//Gauss quadrature weights and points
    double wg[2],sg[2];
    wg[0] = 1.0;
    wg[1] = 1.0;
    sg[0] = -0.577350269189;
    sg[1] = 0.577350269189;

    //first block row
    for(int ii=0;ii<mz;ii++)
    {
        for (int jj=0;jj<mz;jj++)
        {           
//            printf("\n");
           // printf(" %d, %d ",ii,jj);
            Aglobal[ii][jj]=Aglobal[ii][jj]+Amat[ii][jj]+F0mat[ii][jj]+Cmat[ii][jj];
            je=mz+jj; 
  
           // printf(" %d, %d ",ii,je);
            Aglobal[ii][je]=Aglobal[ii][je]+Dmat[ii][jj];
        }
    }





/*
    for(int ii=0;ii<mz;ii++)
    {
        printf("\n");
        for (int jj=0;jj<mz;jj++)
        {
           
            printf(" %f ", Aglobal[ii][jj]);
        }
    }
*/

//compute right-hand side
    rhsglobal[0]=nel*penal;
    rhsglobal[1]=nel*penal*(-1.0)-ss*2.0*nel;
    rhsglobal[2]=nel*penal+ss*4*nel;

    for(int ig=0;ig<2;ig++)
    {
        rhsglobal[0]=rhsglobal[0]+wg[ig]*sourcef((sg[ig]+1)/(2*nel))/(2*nel);
        rhsglobal[1]=rhsglobal[1]+wg[ig]*sg[ig]*sourcef((sg[ig]+1)/(2*nel))/(2*nel);
        rhsglobal[2]=rhsglobal[2]+wg[ig]*sg[ig]*sg[ig]*sourcef((sg[ig]+1)/(2*nel))/(2*nel);
        
    }
//printf(" %d %f ",3 , rhsglobal[3]);
// for (int jj=0;jj<mz;jj++)
//        {
//            printf(" %f ", rhsglobal[jj]);
//        }

//intermediate block rows
//loop over elements
    for (int i=1;i<nel-1;i++)
    {
        for(int ii=0;ii<mz;ii++)
        {   
            ie=ii+(i)*mz;
//            printf(" %d \n",ie);
            for(int jj=0;jj<mz;jj++)
	    {
		//je=jj+(i-1)*mz;
                je=jj+(i)*mz;
                printf("i= %d ,j= %d\n ",ie, je);
		Aglobal[ie][je]=Aglobal[ie][je]+Amat[ii][jj]+Bmat[ii][jj]+Cmat[ii][jj];

                //printf(" %d, %f ",je, Aglobal[ie][je]);
		//je=jj+(i-2)*mz;
		je=jj+(i-1)*mz;
                printf("i= %d ,j= %d \n",ie, je);
                Aglobal[ie][je]=Aglobal[ie][je]+Emat[ii][jj];
		//je=jj+(i)*mz;

		je=jj+(i+1)*mz;
//                printf(" %d , %d, %d, %d, %d ",ie, je, jj,i,mz);
                printf("i= %d ,j= %d \n",ie, je);  
                Aglobal[ie][je]=Aglobal[ie][je]+Dmat[ii][jj];
                printf("\n\n");
             //   printf(" %d , %d ,%f ",ie, je, Aglobal[ie][je]);
	    }

//compute right-hand side
            for(int ig=0;ig<2;ig++)
            {   
                
                double a=pow(sg[ig],ii);
               // printf("ig=%d sg[ig]=%f ii=%d a=%f",ig, sg[ig],ii+1,a);
                double b=sourcef((sg[ig]+2*(i)+1.0)/(2*nel))/(2*nel);
             // printf("ig=%d, sg[ig]= %f,i= %d nel=%d, 2*nel= %d, b= %f",ig,sg[ig],i,nel,2*nel,b);
                rhsglobal[ie]=rhsglobal[ie]+wg[ig]*a*b;            
              //  rhsglobal[ie]=rhsglobal[ie]+wg[ig]*(pow(sg[ig],(ii-1)))*sourcef((sg[ig]+2*(i-1)+1.0)/(2*nel))/(2*nel);
//                printf("\n");
//                printf("ie=%d, ii=%d, wg[ig]=%f, sg[ig]=%f, a=%f,b=%f, rhsglobal[ie]=%f",ie ,ii, wg[ig],sg[ig], a,b, rhsglobal[ie]);
            }
        }       
    }

//printf("s=%f",sourcef(1.0));

   //  printf("\n");

//last block row
printf("Last block");
    for(int ii=0;ii<mz;ii++)
    {
        ie=ii+(nel-1)*mz;
        for(int jj=0;jj<mz;jj++)
        {
            printf("\n\n");
            je=jj+(nel-1)*mz;
            printf("i= %d ,j= %d \n",ie, je);
            if((ie==11)&&(je==11))
            {
                printf("ie=%d, je=%d, Aglobal=%f, ii=%d,jj=%d, Amat=%f, FNmat=%f, Bmat=%f",ie, je, Aglobal[ie][je],ii-1,jj-1,Amat[ii-1][jj-1],FNmat[ii-1][jj-1],Bmat[ii-1][jj-1]); 
            }
            Aglobal[ie][je]=Aglobal[ie][je]+Amat[ii][jj]+FNmat[ii][jj]+Bmat[ii][jj];
            je=jj+(nel-2)*mz;
            printf("i= %d ,j= %d \n",ie, je);
            Aglobal[ie][je]=Aglobal[ie][je]+Emat[ii][jj];
            //printf("%f", Aglobal[ie][je]);  

      }
        for(int ig=0;ig<2;ig++)
        {
     //    printf("\n");
         double c=(pow(sg[ig],(ii)));
         double d=sourcef((sg[ig]+2*(nel-1)+1.0)/(2*nel))/(2.0*nel); 
        // printf("ig=%d, sg= %f, ii= %d, c=%f, d=%f",ig,sg[ig],ii,c,d); 
         rhsglobal[ie]=rhsglobal[ie]+wg[ig]*c*d;
         //printf("%d, %f, %f, %f \n ",ie,c,d,rhsglobal[ie]);
         printf("\n");
        }
    }

/*
for(int ii=0;ii<mz;ii++)
  {
        ie=ii+(nel-1)*mz;
        for(int jj=0;jj<mz;jj++)
        {
            je=jj+(nel-2)*mz;
            Aglobal[ie][je]=Aglobal[ie][je]+Emat[ii][jj];
        }
    }
*/





    for(int i=0;i<glodim;i++)
       {
        fprintf(f,"\n");
        for(int j=0;j<glodim;j++)
            {
             fprintf(f," %f ",Aglobal[i][j]);
            }
       }

//fprintf(f, "Integer: %d, float: %f\n", i, py);

for(int i=0;i<glodim;i++)
    {
         fprintf(g," %f \n",rhsglobal[i]);
    }




    return 0;
}

  


