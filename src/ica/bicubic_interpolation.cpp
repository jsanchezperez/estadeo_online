// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2015-2018, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// All rights reserved.


#include "bicubic_interpolation.h"
#include "transformation.h"

using namespace std;

/**
  *
  * Bicubic interpolation in one dimension
  *
**/
float cubic_interpolation(
  float v[4],  //interpolation points
  float x      //point to be interpolated
)
{
  return v[1]+0.5*x*(v[2]-v[0]
                     +x*(2.0*v[0]-5.0*v[1]+4.0*v[2]-v[3]
                         +x*(3.0*(v[1]-v[2])+v[3]-v[0])));
}


/**
  *
  * Bicubic interpolation in two dimension
  *
**/
float bicubic_interpolation(
  float p[4][4], //array containing the interpolation points
  float x,       //x position to be interpolated
  float y        //y position to be interpolated
)
{
  float v[4];
  v[0]=cubic_interpolation(p[0], y);
  v[1]=cubic_interpolation(p[1], y);
  v[2]=cubic_interpolation(p[2], y);
  v[3]=cubic_interpolation(p[3], y);
  return cubic_interpolation(v, x);
}


/**
  *
  * Compute the bicubic interpolation of a point in an image. 
  * Detects if the point goes outside the image domain
  *
**/
float bicubic_interpolation(
  float *input, //image to be interpolated
  float uu,     //x component of the vector field
  float vv,     //y component of the vector field
  int nx,       //width of the image
  int ny        //height of the image
)
{
  if(uu>=nx-2 || uu<2 || vv>=ny-2 || vv<2) 
    return 999999.9;
  else {
    int sx=(uu<0)? -1: 1;
    int sy=(vv<0)? -1: 1;
    int x, y, mx, my, dx, dy, ddx, ddy;
  
    x  =(int) uu;
    y  =(int) vv;
    mx =(int) uu-sx;
    my =(int) vv-sy;
    dx =(int) uu+sx;
    dy =(int) vv+sy;
    ddx=(int) uu+2*sx;
    ddy=(int) vv+2*sy;

    //obtain the interpolation points of the image
    float p11=input[mx +nx*my];
    float p12=input[x  +nx*my];
    float p13=input[dx +nx*my];
    float p14=input[ddx+nx*my];

    float p21=input[mx +nx*y];
    float p22=input[x  +nx*y];
    float p23=input[dx +nx*y];
    float p24=input[ddx+nx*y];

    float p31=input[mx +nx*dy];
    float p32=input[x  +nx*dy];
    float p33=input[dx +nx*dy];
    float p34=input[ddx+nx*dy];

    float p41=input[mx +nx*ddy];
    float p42=input[x  +nx*ddy];
    float p43=input[dx +nx*ddy];
    float p44=input[ddx+nx*ddy];

    //create array
    float pol[4][4] = { 
      {p11, p21, p31, p41}, {p12, p22, p32, p42},
      {p13, p23, p33, p43}, {p14, p24, p34, p44}
    };

    //return interpolation
    return bicubic_interpolation (pol, (float) uu-x, (float) vv-y);
  }
}


/**
  *
  * Compute the bicubic interpolation of an image from a parametric trasform
  *
**/
void bicubic_interpolation(
  float *input,   //image to be warped
  vector<int> &p, //selected points
  float *output,  //warped output image with bicubic interpolation
  float *params,  //x component of the vector field
  int nparams,    //number of parameters of the transform
  int nx,         //width of the image
  int ny          //height of the image 
)
{
  #pragma omp parallel for
  for (unsigned int i=0; i<p.size(); i++)
  {
    float x, y;
    float x1=p[i]%nx;
    float y1=(int)(p[i]/nx);

    //transform coordinates using the parametric model
    project(x1, y1, params, x, y, nparams);
    
    //obtain the bicubic interpolation at position (x, y)
    output[i]=bicubic_interpolation(input, x, y, nx, ny);
  }
}


/**
  *
  * Compute the bicubic interpolation of an image from a parametric trasform
  *
**/
void bicubic_interpolation(
  float *input,   //image to be warped
  float *output,  //warped output image with bicubic interpolation
  float *params,  //x component of the vector field
  int   nparams,  //number of parameters of the transform
  int   nx,       //width of the image
  int   ny        //height of the image 
)
{
  #pragma omp parallel for
  for(int i=0; i<ny; i++)
    for(int j=0; j<nx; j++)
    {
      float x, y;

      //transform coordinates using the parametric model
      project(j, i, params, x, y, nparams);
      
      //obtain the bicubic interpolation at position (x, y)
      output[i*nx+j]=bicubic_interpolation(input, x, y, nx, ny);
    }
}


/**
  *
  * Function to warp the image using bilinear interpolation
  *
**/
void bilinear_interpolation(
  float *input,   //image to be warped
  vector<int> &p, //selected points
  float *output,  //warped output image with bicubic interpolation
  float *params,  //x component of the vector field
  int nparams,    //number of parameters of the transform
  int nx,         //width of the image
  int ny          //height of the image
)
{
  #pragma omp parallel for
  for (unsigned int i=0; i<p.size(); i++)
  {
    float x1=p[i]%nx;
    float y1=(int)(p[i]/nx);
    float uu, vv;

    //transform coordinates using the parametric model
    project(x1, y1, params, uu, vv, nparams);

    if(uu<1 || uu>nx-2 || vv<1 || vv>ny-2)
      output[i]=999999.9;
    else {
      int sx=(uu<0)? -1: 1;
      int sy=(vv<0)? -1: 1;
      int x, y, dx, dy;

      x =(int) uu;
      y =(int) vv;
      dx=(int) uu+sx;
      dy=(int) vv+sy;
      
      float p1=input[x +nx*y];
      float p2=input[dx+nx*y];
      float p3=input[x +nx*dy];
      float p4=input[dx+nx*dy];

      float e1=((float) sx*(uu-x));
      float E1=((float) 1.0-e1);
      float e2=((float) sy*(vv-y));
      float E2=((float) 1.0-e2);

      float w1=E1*p1+e1*p2;
      float w2=E1*p3+e1*p4;

      output[i]=E2*w1+e2*w2;
    }
  }
}


