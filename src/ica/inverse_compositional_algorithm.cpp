// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2015-2018, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

/** 
  * 
  *  This code implements the 'inverse compositional algorithm' proposed in
  *     [1] S. Baker, and I. Matthews. (2004). Lucas-kanade 20 years on: A 
  *         unifying framework. International Journal of Computer Vision, 
  *         56(3), 221-255.
  *     [2] S. Baker, R. Gross, I. Matthews, and T. Ishikawa. (2004). 
  *         Lucas-kanade 20 years on: A unifying framework: Part 2. 
  *         International Journal of Computer Vision, 56(3), 221-255.
  *  
  *  This implementation is for graylevel images. It calculates the global 
  *  transform between two images. It uses robust error functions and a 
  *  coarse-to-fine strategy for computing large displacements
  * 
**/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "bicubic_interpolation.h"
#include "inverse_compositional_algorithm.h"
#include "matrix.h"
#include "mask.h"
#include "transformation.h"
#include "zoom.h"
#include "file.h"


using namespace std;


/**
 *
 *  Derivative of robust error functions
 *
 */
inline float rhop(
  float t2,     //squared difference of both images  
  float lambda  //robust threshold
)
{
  float result=0.0;
  float lambda2=lambda*lambda;

  //TRUNCATED_QUADRATIC:
  //if(t2<lambda2) result=1.0;
  //else result=0.0;
  
  //LORENTZIAN:
  result=1/(lambda2+t2);
  
  return result;
}

 
/**
 *
 *  Function to compute DI^t*J
 *  from the gradient of the image and the Jacobian
 *
 */
void steepest_descent_images
(
  float *Ix,   //x derivate of the image
  float *Iy,   //y derivate of the image
  float *J,    //Jacobian matrix
  float *DIJ,  //output DI^t*J
  int nparams, //number of parameters
  int N        //number of points
)
{
  #pragma omp parallel for
  for(unsigned int p=0; p<(unsigned int)N; p++)
    for(int n=0; n<nparams; n++)
      DIJ[p*nparams+n]=Ix[p]*J[2*p*nparams+n]+
                       Iy[p]*J[2*p*nparams+n+nparams];
}

/**
 *
 *  Function to compute the Hessian matrix
 *  the Hessian is equal to DIJ^t*DIJ
 *
 */
void hessian
(
  float *DIJ,  //the steepest descent image
  float *H,    //output Hessian matrix
  int nparams, //number of parameters
  int N        //number of values
) 
{
  //initialize the hessian to zero
  #pragma omp parallel for
  for(int k=0; k<nparams*nparams; k++)
    H[k] = 0;
 
  //calculate the hessian in a neighbor window
  #pragma omp parallel for
  for(int k=0; k<nparams; k++)
    for(int l=0; l<nparams; l++)
      for(int i=0; i<N; i++)
	     H[k*nparams+l]+=DIJ[i*nparams+k]*DIJ[i*nparams+l];
}


/**
 *
 *  Function to compute the Hessian matrix with robust error functions
 *  the Hessian is equal to rho'*DIJ^t*DIJ
 *
 */
void hessian
(
  float *DIJ,  //the steepest descent image
  float *rho,  //robust function
  float *H,    //output Hessian matrix
  int nparams, //number of parameters
  int N        //number of values
) 
{
  //initialize the hessian to zero  
  #pragma omp parallel for
  for(int k=0; k<nparams*nparams; k++)
    H[k]=0;

  //calculate the hessian in a neighbor window
  #pragma omp parallel for
  for(int k=0; k<nparams; k++)
    for(int l=0; l<nparams; l++)
      for(int i=0; i<N; i++)
	     H[k*nparams+l]+=rho[i]*DIJ[i*nparams+k]*DIJ[i*nparams+l];
}



/**
 *
 *  Function to compute the inverse of the Hessian
 *
 */
void inverse_hessian
(
  float *H,   //input Hessian
  float *H_1, //output inverse Hessian 
  int nparams //number of parameters
) 
{
  if(inverse(H, H_1, nparams)==-1) 
    //if the matrix is not invertible, set parameters to infinity
    for(int i=0; i<nparams*nparams; i++) H_1[i]=999999.9;
}


/**
 *
 *  Function to compute I2(W(x;p))-I1(x)
 *
 */
void difference_image
(
  float *I,  //first image I1(x)
  float *Iw, //second warped image I2(x'(x;p)) 
  float *DI, //output difference array
  vector<int> &x //points
) 
{
  #pragma omp parallel for
  for(unsigned int i=0; i<x.size(); i++)
    DI[i]=Iw[i]-I[x[i]];
}


/**
 *
 *  Function to store the values of p'((I2(x'(x;p))-I1(x))²)
 *
 */
void robust_error_function
(
  float *DI,   //input difference array
  float *rho,  //output robust function
  float lambda,//threshold used in the robust functions
  int   N      //number of values
)
{ 
  #pragma omp parallel for
  for(int i=0;i<N;i++)
  {
    float norm=DI[i]*DI[i];
    rho[i]=rhop(norm,lambda);
  }
}


/**
 *
 *  Function to compute b=Sum(DIJ^t * DI)
 *
 */
void independent_vector
(
  float *DIJ, //the steepest descent image
  float *DI,  //I2(x'(x;p))-I1(x) 
  float *b,   //output independent vector
  int   nparams, //number of parameters
  int   N     //number of columns
)
{
  //initialize the vector to zero
  for(int k=0; k<nparams; k++)
    b[k]=0;

  #pragma omp parallel for
  for(int k=0; k<nparams; k++)
    for(int i=0; i<N; i++)
      b[k]+=DIJ[i*nparams+k]*DI[i];
}


/**
 *
 *  Function to compute b=Sum(rho'*DIJ^t * DI)
 *  with robust error functions
 *
 */
void independent_vector
(
  float *DIJ,    //the steepest descent image
  float *DI,     //I2(x'(x;p))-I1(x) 
  float *rho,    //robust function
  float *b,      //output independent vector
  int   nparams, //number of parameters
  int   N        //number of values
)
{
  //initialize the vector to zero
  for(int k=0; k<nparams; k++)
    b[k]=0;

  #pragma omp parallel for
  for(int k=0; k<nparams; k++)
    for(int i=0; i<N; i++)
      b[k]+=rho[i]*DIJ[i*nparams+k]*DI[i];
}


/**
 *
 *  Function to solve for dp
 *  
 */
float parametric_solve
(
  float *H_1, //inverse Hessian
  float *b,   //independent vector
  float *dp,  //output parameters increment 
  int nparams //number of parameters
)
{
  float error=0.0;
  Axb(H_1, b, dp, nparams);
  for(int i=0; i<nparams; i++) error+=dp[i]*dp[i];
  return sqrt(error);
}


/**
  *
  *  Select points
  *
**/
void select_points(
  vector<int> &x,
  int nx,
  int ny
)
{
  if(nx>64)
  {
    int radius=5;
    int region=5;
    int border=(float)(nx/10.);
    for(int i=border+radius;i<ny-border-radius;i+=radius*radius)
    for(int j=border+radius;j<nx-border-radius;j+=radius*radius)
    {
      for(int k=i-region;k<=i+region; k++)
        for(int l=j-region;l<=j+region; l++)
          x.push_back(k*nx+l);
    } 
  }
  else
    for(int k=8;k<ny-8; k++)
        for(int l=8;l<nx-8; l++)
          x.push_back(k*nx+l);	
}


/**
  *
  *  Inverse compositional algorithm
  *  Quadratic version - L2 norm
  * 
  *
**/
void inverse_compositional_algorithm(
  float *I1,   //first image
  float *I2,   //second image
  float *p,    //parameters of the transform (output)
  int nparams, //number of parameters of the transform
  float TOL,   //Tolerance used for the convergence in the iterations
  int   nx,    //number of columns
  int   ny     //number of rows
)
{
  //find corner points
  vector<int> x;
  select_points(x, nx, ny);
  
  int N=x.size();
  int size2=N*nparams;       //size of the image with transform parameters
  int size3=nparams*nparams; //size for the Hessian
  int size4=2*N*nparams; 
  
  float *Iw=new float[N];      //warp of the second image/
  float *DI=new float[N];      //error image (I2(w)-I1)
  float *DIJ=new float[size2]; //steepest descent images
  float *dp=new float[nparams];//incremental solution
  float *b=new float[nparams]; //steepest descent images
  float *J=new float[size4];   //jacobian matrix for all points
  float *H=new float[size3];   //Hessian matrix
  float *H_1=new float[size3]; //inverse Hessian matrix
  float *Ix=new float[N];      //x derivate of the first image
  float *Iy=new float[N];      //y derivate of the first image

  //Evaluate the gradient of I1
  gradient(I1, Ix, Iy, x, nx);

  //Evaluate the Jacobian
  jacobian(J, x, nparams, nx);

  //Compute the steepest descent images
  steepest_descent_images(Ix, Iy, J, DIJ, nparams, N);

  //Compute the Hessian matrix
  hessian(DIJ, H, nparams, N);
  inverse_hessian(H, H_1, nparams);

  //Iterate
  float error=1E10;
  int niter=0;

  do{     
    //Warp image I2
    //bicubic_interpolation(I2, x, Iw, p, nparams, nx, ny);
    bilinear_interpolation(I2, x, Iw, p, nparams, nx, ny);

    //Compute the error image (I1-I2w)
    difference_image(I1, Iw, DI, x);
    
    //Compute the independent vector
    independent_vector(DIJ, DI, b, nparams, N);

    //Solve equation and compute increment of the motion 
    error=parametric_solve(H_1, b, dp, nparams);

    //Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
    update_transform(p, dp, nparams);

    niter++;    
  }
  while(error>TOL && niter<MAX_ITER);

  delete []Iw;
  delete []DI;
  delete []DIJ;
  delete []dp;
  delete []b;
  delete []J;
  delete []H;
  delete []H_1;
  delete []Ix;
  delete []Iy;
}



/**
  *
  *  Inverse compositional algorithm 
  *  Version with robust error functions
  * 
**/
void robust_inverse_compositional_algorithm(
  float *I1,     //first image
  float *I2,     //second image
  float *p,      //parameters of the transform (output)
  int   nparams, //number of parameters of the transform
  float TOL,     //Tolerance used for the convergence in the iterations
  float lambda,  //parameter of robust error function
  int   nx,      //number of columns
  int   ny       //number of rows
)
{  
  //find reference points
  vector<int> x;
  select_points(x, nx, ny);      

  int N=x.size();              //number of corner points
  int size2=N*nparams;         //size of the image with transform parameters
  int size3=nparams*nparams;   //size for the Hessian
  int size4=2*N*nparams; 
  
  float *Iw=new float[N];      //warp of the second image/
  float *DI=new float[N];      //error image (I2(w)-I1)
  float *DIJ=new float[size2]; //steepest descent images
  float *dp=new float[nparams];//incremental solution
  float *b=new float[nparams]; //steepest descent images
  float *J=new float[size4];   //jacobian matrix for all points
  float *H=new float[size3];   //Hessian matrix
  float *H_1=new float[size3]; //inverse Hessian matrix
  float *Ix=new float[N];      //x derivate of the first image
  float *Iy=new float[N];      //y derivate of the first image
  float *rho=new float[N];     //robust function  

  //Evaluate the gradient of I1
  gradient(I1, Ix, Iy, x, nx);

  //Evaluate the Jacobian
  jacobian(J, x, nparams, nx);

  //Compute the steepest descent images
  steepest_descent_images(Ix, Iy, J, DIJ, nparams, N);
  
  //Iterate
  float error=1E10;
  int niter=0;
  float lambda_it;
  
  if(lambda>0) lambda_it=lambda;
  else lambda_it=LAMBDA_0;
  
  do{     
    //Warp image I2
    //bicubic_interpolation(I2, x, Iw, p, nparams, nx, ny);
    bilinear_interpolation(I2, x, Iw, p, nparams, nx, ny);

    //Compute the error image (I1-I2w)
    difference_image(I1, Iw, DI, x);

    //compute robustifiction function
    robust_error_function(DI, rho, lambda_it, N);
    if(lambda<=0 && lambda_it>LAMBDA_N) 
    {
      lambda_it*=LAMBDA_RATIO;
      if(lambda_it<LAMBDA_N) lambda_it=LAMBDA_N;
    }

    //Compute the independent vector
    independent_vector(DIJ, DI, rho, b, nparams, N);

    //Compute the Hessian matrix
    hessian(DIJ, rho, H, nparams, N);
    inverse_hessian(H, H_1, nparams);

    //Solve equation and compute increment of the motion 
    error=parametric_solve(H_1, b, dp, nparams);

    //Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
    update_transform(p, dp, nparams);

    niter++;    
  }
  while(error>TOL && niter<MAX_ITER);

  delete []Iw;
  delete []DI;
  delete []DIJ;
  delete []dp;
  delete []b;
  delete []J;
  delete []H;
  delete []H_1;
  delete []Ix;
  delete []Iy;
  delete []rho;
}


/**
  *
  *  Multiscale approach for computing the optical flow
  *
**/
void pyramidal_inverse_compositional_algorithm(
    float *I1,     //first image
    float *I2,     //second image
    float *p,      //parameters of the transform
    int   nparams, //number of parameters
    int   nxx,     //image width
    int   nyy,     //image height
    int   cscale,  //coarsest scale
    int   fscale,  //finest scale 
    float TOL,     //stopping criterion threshold
    int   robust,  //robust error function
    float lambda   //parameter of robust error function
)
{
    int size=nxx*nyy;

    float **I1s=new float*[cscale];
    float **I2s=new float*[cscale];
    float **ps =new float*[cscale];

    int *nx=new int[cscale];
    int *ny=new int[cscale];

    I1s[0]=new float[size];
    I2s[0]=new float[size];

    //copy the input images
    #pragma omp parallel for
    for(int i=0;i<size;i++)
    {
      I1s[0][i]=I1[i];
      I2s[0][i]=I2[i];
    }

    ps[0]=p;
    nx[0]=nxx;
    ny[0]=nyy;

    //initialization of the transformation parameters at the finest scale
    for(int i=0; i<nparams; i++)
      p[i]=0.0;

    //create the scales
    for(int s=1; s<cscale; s++)
    {
      zoom_size(nx[s-1], ny[s-1], nx[s], ny[s]);

      const int size=nx[s]*ny[s];

      I1s[s]=new float[size];
      I2s[s]=new float[size];
      ps[s] =new float[nparams];
      
      for(int i=0; i<nparams; i++)
        ps[s][i]=0.0;

      //zoom the images from the previous scale
      zoom_out(I1s[s-1], I1s[s], nx[s-1], ny[s-1]);
      zoom_out(I2s[s-1], I2s[s], nx[s-1], ny[s-1]);
    }  

    //pyramidal approach for computing the transformation
    for(int s=cscale-1; s>=0; s--)
    {
      //compute transformation for maximum numer of scales
      if(s>=fscale-1)
      {
        //incremental refinement for this scale
        if(robust==QUADRATIC)
          inverse_compositional_algorithm(
            I1s[s], I2s[s], ps[s], nparams, TOL, nx[s], ny[s]
          );
        else
          robust_inverse_compositional_algorithm(
            I1s[s], I2s[s], ps[s], nparams, TOL, 
            lambda, nx[s], ny[s]
          );
      }
      
      //if it is not the finest scale, then upsample the parameters
      if(s) 
        zoom_in_parameters(
          ps[s], ps[s-1], nparams, nx[s], ny[s], nx[s-1], ny[s-1]
        );
    }

    //delete allocated memory
    delete []I1s[0];
    delete []I2s[0];
    for(int i=1; i<cscale; i++)
    {
      delete []I1s[i];
      delete []I2s[i];
      delete []ps [i];
    }
    delete []I1s;
    delete []I2s;
    delete []ps;
    delete []nx;
    delete []ny;
}
