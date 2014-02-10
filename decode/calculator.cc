/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

* ./calculator.cc
*	FILE: cluster.cc 				                                *
*										                            *
*   				      				                            *
*   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>					*
*   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>

#include "calculator.h"

using namespace std;

Calculator::Calculator() {
}

float Calculator::sum_logs(vector<float> log_reg) {
   float marginal_max = find_log_max(log_reg);
   double marginal_sum = 0;
   for (unsigned int i = 0; i < log_reg.size(); ++i) {
      marginal_sum += exp(log_reg[i] - marginal_max);
   }
   return (marginal_max + (float) log(marginal_sum));
}

float Calculator::find_log_max(vector<float> log_reg){
   float max = log_reg[0]; 
   for (unsigned int i = 1; i < log_reg.size(); ++i) { 
      if (log_reg[i] > max) {
         max = log_reg[i]; 
      }
   }
   return max;
}

Calculator::~Calculator() {
}
