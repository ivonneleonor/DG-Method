/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

const char *sSDKname     = "conjugateGradient";


float sourcef(float xval)
    {
        //source function for exact solution = (1-x)e^(-x^2)
         float yval;
         yval=-(2*xval-2*(1-2*xval)+4*xval*(xval-pow(xval,2)))*exp(-xval*xval);

        return yval;
    }



/* genTridiag: generate a random tridiagonal symmetric matrix 
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

*/

int main(int argc, char **argv)
{
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    int *Ib = NULL, *Jb = NULL;
  //  float *val = NULL;
    float *valb = NULL;
    const float tol = 1e-5f;
    const int max_iter = 10000;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
  //  int *d_col, *d_row;
    int *d_colb, *d_rowb;
 //   float *d_val;
    float *d_x, dot;
    float *d_valb;
    float *d_r, *d_p, *d_Ax;
    int k,nnzb=0;
    float alpha, beta, alpham1;
    int *nnzPerRowColumn;
    int nnzTotalDevHostPtr;
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);


    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

/* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);



 /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);






//////////////////////////Riviere code//////////////////////////////////////////
 
//number of subintervals
    const int nel=4,mz=3;

    int glodim,je,ie;
    float ss,penal;
    float Amat[3][3],Bmat[3][3],Cmat[3][3],Dmat[3][3],Emat[3][3],F0mat[3][3],FNmat[3][3];

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
    //nonsymmetric interior penalty Galerkin (NIPG) method
    ss=1.0;
    penal=1.0;

    //dimension of global matrix
    glodim = nel * mz;
    float Aglobal[glodim][glodim];
    float rhsglobal[glodim];

    int rows = glodim, cols = glodim;

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
        for(int i=0;i<mz;i++)
        {
            Amat[i][j]=nel*Amat[i][j];
        }
    }
    
    Bmat[0][0]=penal;
    Bmat[0][1]=1.0-penal;
    Bmat[0][2]=-2.0+penal;
    Bmat[1][0]=-ss-penal;
    Bmat[1][1]=-1.0+ss+penal;
    Bmat[1][2]=2.0-ss-penal;
    Bmat[2][0]=2.0*ss+penal;
    Bmat[2][1]=1.0-2.0*ss-penal;
    Bmat[2][2]=-2.0+2.0*ss+penal;


    for(int i=0;i<mz;i++)
    {  
        for(int j=0;j<mz;j++)
        {
            Bmat[i][j]=nel*Bmat[i][j];
        }
    }

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
        for(int j=0;j<mz;j++)
        {
            Cmat[i][j]=nel*Cmat[i][j];
        }
    }

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
        for(int j=0;j<mz;j++)
        {
            Dmat[i][j]=nel*Dmat[i][j];
        }
    }

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
        for(int j=0;j<mz;j++)
        {
            Emat[i][j]=nel*Emat[i][j];
        }
    }

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
        for(int j=0;j<mz;j++)
        {
            F0mat[i][j]=nel*F0mat[i][j];
        }
    }

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
        for(int j=0;j<mz;j++)
        {
            FNmat[i][j]=nel*FNmat[i][j];
        }
    }

//Gauss quadrature weights and points
    float wg[2],sg[2];
    wg[0] = 1.0;
    wg[1] = 1.0;
    sg[0] = -0.577350269189;
    sg[1] = 0.577350269189;

    //first block row
    for(int ii=0;ii<mz;ii++)
    {
        for (int jj=0;jj<mz;jj++)
        {           
            Aglobal[ii][jj]=Aglobal[ii][jj]+Amat[ii][jj]+F0mat[ii][jj]+Cmat[ii][jj];
            je=mz+jj; 
            Aglobal[ii][je]=Aglobal[ii][je]+Dmat[ii][jj];
        }
    }

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
    
for (int i=1;i<nel-1;i++)
    {
        for(int ii=0;ii<mz;ii++)
        {   
            ie=ii+(i)*mz;
            for(int jj=0;jj<mz;jj++)
	    {
                je=jj+(i)*mz;
		Aglobal[ie][je]=Aglobal[ie][je]+Amat[ii][jj]+Bmat[ii][jj]+Cmat[ii][jj];

		je=jj+(i-1)*mz;
                Aglobal[ie][je]=Aglobal[ie][je]+Emat[ii][jj];

		je=jj+(i+1)*mz;
                Aglobal[ie][je]=Aglobal[ie][je]+Dmat[ii][jj];
	    }

//compute right-hand side
            for(int ig=0;ig<2;ig++)
            {   
                
                float a=pow(sg[ig],ii);
                float b=sourcef((sg[ig]+2*(i)+1.0)/(2*nel))/(2*nel);
                rhsglobal[ie]=rhsglobal[ie]+wg[ig]*a*b;            
            }
        }       
    }


    for(int ii=0;ii<mz;ii++)
    {
        ie=ii+(nel-1)*mz;
        for(int jj=0;jj<mz;jj++)
        {
            je=jj+(nel-1)*mz;
            Aglobal[ie][je]=Aglobal[ie][je]+Amat[ii][jj]+FNmat[ii][jj]+Bmat[ii][jj];
            je=jj+(nel-2)*mz;
            Aglobal[ie][je]=Aglobal[ie][je]+Emat[ii][jj];
      }
        for(int ig=0;ig<2;ig++)
        {
         float c=(pow(sg[ig],(ii)));
         float d=sourcef((sg[ig]+2*(nel-1)+1.0)/(2*nel))/(2.0*nel); 
         rhsglobal[ie]=rhsglobal[ie]+wg[ig]*c*d;
        }
    }

    for(int i=0;i<glodim;i++)
       {
        fprintf(f,"\n");
        for(int j=0;j<glodim;j++)
            {
             fprintf(f," %f ",Aglobal[i][j]);
            }
       }

for(int i=0;i<glodim;i++)
    {
         fprintf(g," %f \n",rhsglobal[i]);
    }

   

////////////////////finish Riviere code////////////////////////////////777/////

//implement matrix Aglobal to csr line 443
 
   float *Bglobal = NULL,*d_Bglobal=NULL;
   Bglobal = (float *)malloc(sizeof(float)*((glodim+1)*(glodim+1)));

   for(int i=0;i<glodim;i++)
       {
        printf("\n");
        for(int j=0;j<glodim;j++)
            {
               Bglobal[(glodim+1)*i+j]=Aglobal[i][j];
               printf("Bglobal[%d]=%f \n",(glodim+1)*i+j,Bglobal[(glodim+1)*i+j]);
            }
       }

   cudaMalloc((void **)&d_Bglobal, sizeof(float)* (glodim + 1)*(glodim + 1));
   cudaMemcpy(d_Bglobal,Bglobal,sizeof(float) * (glodim + 1)*(glodim + 1),cudaMemcpyHostToDevice);
 
 

   /* Generate a random tridiagonal symmetric matrix in CSR format */
//    M = N = 1048576;
//    nz = (N-2)*3 + 4;
//    I = (int *)malloc(sizeof(int)*(N+1));
//    J = (int *)malloc(sizeof(int)*nz);
//    val = (float *)malloc(sizeof(float)*nz);
//    genTridiag(I, J, val, N, nz);

    x = (float *)malloc(sizeof(float)*glodim);
    rhs = (float *)malloc(sizeof(float)*glodim);

    for (int i = 0; i < glodim; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }
//allocating values from  Aglobal
    nnzb = (glodim-2)*3 + 4;
    Ib = (int *)malloc(sizeof(int)*(glodim+1));
    Jb = (int *)malloc(sizeof(int)*nnzb);
    valb = (float *)malloc(sizeof(float)*nnzb);



/* Get handle to the CUSPARSE context 
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
*/

//calculate number of zeros pero row and per column 483
    cudaMalloc((void **)&nnzPerRowColumn, sizeof(int)* glodim);
    cusparseSnnz(cusparseHandle,CUSPARSE_DIRECTION_COLUMN,glodim,glodim,descr,d_Bglobal,glodim, nnzPerRowColumn, &nnzTotalDevHostPtr);

//    printf("nnzPerRowColumn=%d\n",nnzPerRowColumn);
    printf("nnzTotalDevHostPtr=%d\n",nnzTotalDevHostPtr);

//allocating memory for matrix in csr from for Aglobal
    checkCudaErrors(cudaMalloc((void **)&d_colb, nnzb*sizeof(int)));
   checkCudaErrors(cudaMalloc((void **)&d_rowb, (glodim+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_valb, nnzb*sizeof(float)));

    cudaMemcpy(d_colb, Jb, nnzb*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowb, Ib, (glodim+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valb, valb, nnzb*sizeof(float), cudaMemcpyHostToDevice);




    cusparseSdense2csr(cusparseHandle, glodim,glodim,descr,d_Bglobal,glodim,nnzPerRowColumn,d_valb,d_rowb,d_colb);

 
// int *d_colb, *d_rowb;
 //   float *d_valb;

  

    checkCudaErrors(cudaMalloc((void **)&d_colb, nnzb*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_rowb, (glodim+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_valb, nnzb*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, glodim*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, glodim*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, glodim*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, glodim*sizeof(float)));




    cudaMemcpy(d_colb, Jb, nnzb*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowb, Ib, (glodim+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valb, valb, nnzb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, glodim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, glodim*sizeof(float), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, glodim, glodim, nnzb, &alpha, descr, d_valb, d_rowb, d_colb, d_x, &beta, d_Ax);

    cublasSaxpy(cublasHandle, glodim, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, glodim, d_r, 1, d_r, 1, &r1);

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, glodim, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, glodim, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, glodim, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, glodim, glodim, nnzb, &alpha, descr, d_valb, d_rowb, d_colb, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, glodim, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, glodim, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, glodim, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, glodim, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaMemcpy(x, d_x, glodim*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = Ib[i]; j < Ib[i+1]; j++)
        {
            rsum += valb[j]*x[Jb[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(Ib);
    free(Jb);
    free(valb);
    free(x);
    free(rhs);
    //free(Bglobal);
    cudaFree(d_colb);
    cudaFree(d_rowb);
    cudaFree(d_valb);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("Test Summary:  Error amount = %f\n", err);
    exit((k <= max_iter) ? 0 : 1);
}
