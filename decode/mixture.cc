/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

* ./mixture.cc
*	FILE: cluster.cc 				                                *
*										                            *
*   				      				                            *
*   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>					*
*   Feb 2014							                            *
*********************************************************************/
#include <cstring>
#include <iostream>
#include "mixture.h"

using namespace std;

Mixture::Mixture(const int s_vector_dim) {
   vector_dim = s_vector_dim;
   mean = new float[vector_dim];
   var = new float[vector_dim];
   for (int i = 0 ; i < vector_dim; ++i) {
      mean[i] = 0;
      var[i] = 0;
   }
}

Mixture::Mixture(const Mixture& src) {
   vector_dim = src.get_vector_dim();
   mean = new float[vector_dim];
   var = new float[vector_dim];
   memcpy(mean, src.get_mean(), sizeof(float) * vector_dim);
   memcpy(var, src.get_var(), sizeof(float) * vector_dim);
}

const Mixture& Mixture::operator= (const Mixture& src) {
   if (this == &src) {
      return *this;
   }
   vector_dim = src.get_vector_dim();
   memcpy(mean, src.get_mean(), sizeof(float) * vector_dim);
   memcpy(var, src.get_var(), sizeof(float) * vector_dim);
   return *this;
}

void Mixture::set_weight(const float s_w) {
   weight = s_w;
}

void Mixture::set_det(const float s_det) {
   det = s_det;
}

void Mixture::set_mean(const float* s_mean) {
   memcpy(mean, s_mean, sizeof(float) * vector_dim);
}

void Mixture::set_var(const float* s_var) {
   memcpy(var, s_var, sizeof(float) * vector_dim);
}

float Mixture::compute_likelihood(const float* data) {
   float exponet = 0.0;
   for(int i = 0; i < vector_dim; ++i) {
      exponet += ((data[i] - mean[i]) * (data[i] - mean[i]) * var[i]);
   }
   exponet /= -2;
   return (det + exponet);
}

Mixture::~Mixture() {
   delete[] mean;
   delete[] var;
}
