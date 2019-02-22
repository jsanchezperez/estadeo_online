#ifndef ESTADEO_H
#define ESTADEO_H

#include "utils.h"


/**
 *
 * Class for online video stabilization
 *
**/
class estadeo {

  public:
  
    estadeo(
      int   np,    //number of parameters of the transformations
      float sigm,  //Gaussian standard deviation for smoothing
      int   verb   //switch on verbose mode
    );
    
    ~estadeo();
    
    void process_frame(
      float *I1,    //input previous grayscale image 
      float *I2,    //input last grayscale image
      float *Ic,    //input last color image to warp
      Timer &timer, //manage runtimes
      int   nx,     //number of columns 
      int   ny,     //number of rows
      int   nz      //number of channels
    );
    
    float *get_H();

    float *get_smooth_H();
    
    int obtain_radius(){return (int)3*sigma;}

  
  private:
  
    void compute_motion(
      float *I1, //first image
      float *I2, //second image
      int   nx,  //number of columns
      int   ny   //number of rows
    );

    void motion_smoothing();
    
    void frame_warping(
      float *I, //frame to be warped
      int   nx, //number of columns   
      int   ny, //number of rows
      int   nz  //number of channels
    );

    
    //function for Gaussian convolution
    void gaussian(int i, int rad, int n);
    
  private:
  
    int   Nf;      //number of frames
    int   Np;      //number of parameters in the transformation
    float sigma;   //Gaussian standard deviation 
    int   radius;  //radius of the Gaussian convolution
    float *Hs;     //last smoothing transform
    float *Hp;     //last stabilizing transform
    int   verbose; //verbose mode
    
    //variables for the circular array
    int   N;    //circular array size
    int   fc;   //current frame position
    float *H;   //original matrix transformation
    float *Hc;  //composition of transformations
    float *H_1; //inverse transformations
};


#endif
