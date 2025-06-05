#include "glsl.h"

#define iChannel0 sampler2D()
#define iChannel1 sampler2D()
#define iChannel2 sampler2D()
#define iChannel3 sampler2D()
#define iTime 0.0f
#define iFrame 0
#define in
#define out
//// https://www.shadertoy.com/view/ld3Gz2
//#include "snail.h"
//vec2 map(vec3 p)
//{
//  vec4 matInfo;
//  vec2 opaque = mapOpaque(p, matInfo);
//  vec2 transparent = mapTransparent(p, matInfo);
//  return opaque.x < transparent.x ? opaque : transparent;
//}
//const float scale = 1.5;
//const vec3 offset(-0.8, 0.0, 0.0);

// https://www.shadertoy.com/view/ld3Gz2
//#include "snail.h"
//vec2 map(vec3 p)
//{
//  vec4 matInfo;
//  vec2 res(1e5,0);
//  // Lines 340-354 from https://www.shadertoy.com/view/ld3Gz2
//#include "just_snail.h"
//  return res;
//}
//const float scale = 0.95;
//const vec3 offset(-0.35, 0.2, 0.0);

//// https://www.shadertoy.com/view/MsXGWr
//#include "mike.h"
//const float scale = 1.65;
//const vec3 offset(0.0, 1.45, 0.0);

// https://www.shadertoy.com/view/WsXSDH
#include "cheeseburger-v2.h"
const float scale = 2.9;
//const vec3 offset(0.0, 0.0, 1.72);
const vec3 offset(0.0, 0.0, 0.0);
vec2 map(vec3 p)
{
  vec2 res;
  res.x = model(p);
  return res;
}

#undef out
#undef in

#include <iostream>
#include <cstdio>

#include <igl/grid.h>
#include <igl/marching_cubes.h>
#include <igl/icosahedron.h>
#include <igl/write_triangle_mesh.h>
#include <igl/parallel_for.h>

int main()
{
  const int ns = 64;
  Eigen::MatrixXd GV;
  igl::grid( Eigen::Vector3i(ns,ns,ns), GV);
  GV.rowwise() -= Eigen::RowVector3d(0.5, 0.5, 0.5);
  GV *= 2.0; 
  // Now grid is [-1,1]³
  
  // Adjust to match example
  GV *= scale;
  GV.rowwise() += Eigen::RowVector3d(offset.x, offset.y, offset.z);
  Eigen::VectorXd S(GV.rows());
  igl::parallel_for(GV.rows(), [&](const int i){
    vec3 p(GV(i, 0), GV(i, 1), GV(i, 2));
    S(i) = map(p).x;
  });

  const bool add_box = true;
  if(add_box)
  {
    // Trick to put box around grid
    for(int i = 0;i<ns;i++)
    {
      for(int j = 0;j<ns;j++)
      {
        for(int k = 0;k<ns;k++)
        {
          // `continue`, unless exactly two of (i,j,k) is either 0 or ns-1
          if( (i==0 || i==ns-1) + (j==0 || j==ns-1) + (k==0 || k==ns-1) != 2 )
            continue;

          int idx = i + j*ns + k*ns*ns;
          S(idx) = -1;
        }
      }
    }
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::marching_cubes(S,GV,ns,ns,ns,0.0,V,F);

  igl::write_triangle_mesh("output.obj", V, F);
}
