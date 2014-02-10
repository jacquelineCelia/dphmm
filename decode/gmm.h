/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

* ./gmm.h
*	FILE: cluster.cc 				                                *
*										                            *
*   				      				                            *
*   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>					*
*   Feb 2014							                            *
*********************************************************************/
#ifndef GMM_H
#define GMM_H

#include <vector>

#include "mixture.h"
#include "calculator.h"

using namespace std;

class Gmm {
   public:
      Gmm(const int, const int);
      void set_mixture_weight(const int, const float);
      void set_mixture_det(const int, const float);
      void set_mixture_mean(const int, const float*);
      void set_mixture_var(const int, const float*);
      void set_tag(const int s_tag) {tag = s_tag;}
      int get_tag() const {return tag;}
      int get_mixture_num() const {return mixture_num;}
      const float* get_mixture_mean(const int);
      const float* get_mixture_var(const int);
      float get_mixture_weight(const int);
      float get_mixture_det(const int);
      float compute_prob(const float*);
      ~Gmm();
   private:
      int mixture_num;
      int vector_dim;
      int tag;
      vector<Mixture> mix;
      Calculator calculator;
};

#endif
