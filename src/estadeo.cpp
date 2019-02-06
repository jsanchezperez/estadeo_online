// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2017-2018, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// All rights reserved.


#include "estadeo.h"
#include "color_bicubic_interpolation.h"
#include "inverse_compositional_algorithm.h"
#include "transformation.h"
#include "matrix.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


estadeo::estadeo(int np, float sigm, int verb): 
         Np(np), sigma(sigm), verbose(verb)
{
  radius=obtain_radius();
  N=(2*radius+1);
  Nf=1;
  fc=0;
 
  //allocate motion transformations for the circular array
  H  =new float[N*Np]; //motion transformations
  Hc =new float[N*Np]; //composition of transformations
  H_1=new float[N*Np]; //inverse transformations

  //introduce identity matrix for the first transform and its inverse
  for(int i=0; i<Np; i++) H[i]=H_1[i]=0;
  
  //allocate last smooth transform
  Hs=new float[Np];
  
  //allocate last stabilizing transform
  Hp=new float[Np];
}

estadeo::~estadeo()
{
  delete []H;
  delete []Hc;
  delete []H_1;
  delete []Hs;
  delete []Hp;
}

/**
  *
  * Main function for Online Video Estabilization
  * Process a frame each time
  * 
**/
void estadeo::process_frame(
  float *I1,    //input previous image of video
  float *I2,    //input last image of video
  float *Ic,    //input last color image to warp
  Timer &timer, //keep runtimes
  int   nx,     //number of columns 
  int   ny,     //number of rows
  int   nz      //number of channels
)
{ 
  //increase the number of frames
  Nf++;
  
  //increase the position of the current frame in the circular array
  fc++;
  if(fc>=N) fc=0;
  
  //step 1. Compute motion between the last two frames
  if(verbose) timer.set_t1();
  compute_motion(I1, I2, nx, ny);
    
  //step 2. Smooth until the last transformation 
  if(verbose) timer.set_t2();
  motion_smoothing();
    
  //step 3. Warp the image
  if(verbose) timer.set_t3();
  frame_warping(Ic, nx, ny, nz);
  if(verbose) timer.set_t4();
}


/**
  *
  * Function for estimating the transformation between two frames
  *
**/
void estadeo::compute_motion
(
  float *I1, //first image
  float *I2, //second image
  int   nx,  //number of columns
  int   ny   //number of rows
)
{
  //parameters for the direct method
  int   nscales=100;
  float TOL=1E-3;
  float lambda=0;
  float N;
  int   robust=LORENTZIAN; //QUADRATIC; //
  int   max_d=200;
  int   min_d=50;

  N=1+log(((nx<ny)?nx:ny)/min_d)/log(2.);
  if ((int) N<nscales) nscales=(int) N;

  //motion estimation through direct methods
  pyramidal_inverse_compositional_algorithm(
    I1, I2, get_H(), Np, nx, ny, nscales, TOL, robust, lambda, max_d
  );
}


/**
  *
  * Function for online motion_smoothing
  * using the local matrix based smoothing approach
  * 
  *
**/
//
//SE PUEDEN CALCULAR DE FORMA INCREMENTAL Hc (con una sola H_1)
void estadeo::motion_smoothing()
{
  //obtain current radius
  int rad=radius; 
  if(rad>=Nf) rad=Nf-1;

  //obtain current size of circular array  
  int n=N;   
  if(n>Nf) n=Nf;

  //compute inverse transform
  inverse_transform(&(H[fc*Np]), &(H_1[fc*Np]), Np);

  //recompute the stabilization for past frames using the circular buffer
  for(int i=Nf-rad; i<Nf; i++)
  {
    //compute backward transformations
    int f=i%n;     //current frame in circular buffer
    int l=(i-1)%n; //previous frame in circular buffer
    
    for(int j=0;j<Np;j++) 
      Hc[l*Np+j]=H_1[f*Np+j];
      
    for(int j=i-2; j>=std::max(i-rad,0); j--)
    {
      int l1=(j+1)%n;
      int l2=j%n;
      compose_transform(&(H_1[l1*Np]), &(Hc[l1*Np]), &(Hc[l2*Np]), Np);
    }

    //introduce the identity matrix in the current frame
    for(int j=0;j<Np;j++) Hc[f*Np+j]=0;

    //compute forward transformations
    if(i<Nf-1)
    { 
      int r=(i+1)%n;
      
      for(int j=0;j<Np;j++) 
        Hc[r*Np+j]=H[r*Np+j];
        
      for(int j=i+2;j<Nf;j++)  //QUITE el <=
      {
        int r1=j%n;
        int r2=(j-1)%n;
        compose_transform(&(H[r1*Np]), &(Hc[r2*Np]), &(Hc[r1*Np]), Np);  
      }
    }

    //convolve with a discrete Gaussian kernel
    gaussian(i, radius, n);

    //compute inverse transformations 
    inverse_transform(Hs, Hp, Np);
  }
}


/**
  *
  * Function for online warping the last frame of the video
  *
**/
void estadeo::frame_warping
(
  float *I, //frame to be warped
  int   nx, //number of columns   
  int   ny, //number of rows
  int   nz  //number of channels
)
{
  int size=nx*ny*nz;

  float *I2=new float[nx*ny*nz];

  //warp the image
  bicubic_interpolation(I, I2, Hp, Np, nx, ny, nz);
  //bilinear_interpolation(I, I2, Hp, Np, nx, ny, nz);

  //copy warped image
  for(int j=0; j<size; j++)
    I[j]=I2[j];

  delete []I2;
}



/**
  *
  * Function to return the motion of the last transformation
  *
**/
float *estadeo::get_H()
{
  return &H[fc*Np];
}


/**
  *
  * Function to compute the motion of the last smooth transformation
  *
**/
float *estadeo::get_smooth_H()
{
  float *H_1=new float[Np];
  float *Htmp=new float[Np];

  inverse_transform(Hp, H_1, Np);
  compose_transform(get_H(), Hp, Htmp, Np);
  compose_transform(H_1, Htmp, Hs, Np);
  
  delete []H_1;
  delete []Htmp;
  
  return Hs;
}


//Gaussian convolution
void estadeo::gaussian(int i, int rad, int n)
{
  //Gaussian convolution in each parameter separately
  for(int p=0; p<Np; p++)
  {
    double average=0.0;
    double sum=0.0;
    
    int j=i-rad;
    
    //left Neumann boundary conditions
    for(; j<=0; j++)
    {
      //apply Gaussian filter using the circular buffer
      double v=Hc[-j*Np+p];
      double norm=0.5*(j-i)*(j-i)/(sigma*sigma);
      double gauss=exp(-norm);
      average+=gauss*v;
      sum+=gauss;
    }
     
    int t=(i+rad>=Nf)? Nf-1: i+rad;
    for(; j<=t; j++)
    {
      //apply Gaussian filter using the circular buffer
      double v=Hc[(j%n)*Np+p];
      double norm=0.5*(j-i)*(j-i)/(sigma*sigma);
      double gauss=exp(-norm);
      average+=gauss*v;
      sum+=gauss;
    }

    //right Neumann boundary conditions
    for(; j<=i+rad; j++)
    {
      //apply Gaussian filter using the circular buffer
      int l=(2*Nf-1-j)%n;
      double v=Hc[l*Np+p];
      double norm=0.5*(j-i)*(j-i)/(sigma*sigma);
      double gauss=exp(-norm);
      average+=gauss*v;
      sum+=gauss;
    }

    Hs[p]=(float) (average/sum);
  }
}

