#include <iostream>

#include "cluster.h"
#include "gmm.h"

using namespace std;

Cluster::Cluster() {
}

void Cluster::init(const int s_state_num, \
                   const int s_mixture_num, \
                   const int s_vector_dim, \
                   const int s_id) {
   state_num = s_state_num;
   mixture_num = s_mixture_num;
   vector_dim = s_vector_dim;
   for (int i = 0 ; i < state_num; ++i) {
      Gmm new_gmm(mixture_num, vector_dim);
      states.push_back(new_gmm);
   }
   for (int i = 0 ; i < state_num; ++i) {
      vector<float> inner_trans;
      for (int j = 0 ; j < state_num + 1; ++j) {
         inner_trans.push_back(0);
      }
      trans.push_back(inner_trans);
   }
   prior = 0;
   id = s_id;
}

void Cluster::set_prior(const float s_prior) {
   prior = s_prior;
}

void Cluster::set_trans(const float* s_trans) {
   for (int i = 0 ; i < state_num; ++i) {
      for (int j = 0 ; j < state_num + 1; ++j) {
         trans[i][j] = s_trans[i * (state_num + 1) + j];
      }
   }
}

void Cluster::set_state_trans(const int i, const int j, const float prob) {
   trans[i][j] = prob;
}

void Cluster::set_state_mixture_weight(const int s, \
  const int m, const float w) {
   states[s].set_mixture_weight(m, w);
}

void Cluster::set_state_mixture_det(const int s, \
  const int m, const float det) {
   states[s].set_mixture_det(m, det); 
}

void Cluster::set_state_mixture_mean(const int s, \
  const int m, const float* mean) {
   states[s].set_mixture_mean(m, mean);
}

void Cluster::set_state_mixture_var(const int s, \
  const int m, const float* var) {
   states[s].set_mixture_var(m, var);
}

float Cluster::compute_state_prob(const int s, const float* frame_i) {
   return states[s].compute_prob(frame_i);
}

const float* Cluster::get_state_mixture_mean(const int s, const int m) {
   return states[s].get_mixture_mean(m);
}

const float* Cluster::get_state_mixture_var(const int s, const int m) {
   return states[s].get_mixture_var(m);
}

float Cluster::get_state_mixture_det(const int s, const int m) {
   return states[s].get_mixture_det(m);
}

float Cluster::get_state_mixture_weight(const int s, const int m) {
   return states[s].get_mixture_weight(m);
}

void Cluster::set_state_tag(const int s, const int tag) {
   states[s].set_tag(tag);
}

int Cluster::get_state_tag(const int s) const {
   return states[s].get_tag();
}

void Cluster::init_prob_table(const int frame_num) {
   for (int i = 0 ; i < state_num; ++i) {
      float prob_arr[frame_num];
      for (int j = 0 ; j < frame_num; ++j) {
         prob_arr[j] = 0.0;
      }
      vector<float> prob(prob_arr, prob_arr + frame_num);
      forward.push_back(prob);
      backward.push_back(prob);
   }
}

void Cluster::set_forward_prob(const int s, const int f, const float p) {
   forward[s][f] = p;
}

void Cluster::set_backward_prob(const int s, const int f, const float p) {
   backward[s][f] = p;
}

void Cluster::show_model() {
   cout << "prior is " << prior << endl;
   cout << "member num is " << member_num << endl;
   for (int s = 0; s < state_num; ++s) {
      for (int n = s; n < state_num; ++n) {
         cout << "trans from " << s  
           << " to " << n << " is " << get_trans_prob(s, n) << endl;
      }
   }
   for (int s = 0; s < state_num; ++s) {
      for (int m = 0; m < states[s].get_mixture_num(); ++m) {
         cout << "state " << s << " mixture " << m << ":" << endl;
         cout << "weight: " << states[s].get_mixture_weight(m) << endl;
         cout << "det: " << states[s].get_mixture_det(m) << endl;
         const float* mean = states[s].get_mixture_mean(m);
         const float* var = states[s].get_mixture_var(m);
         for (int d = 0; d < vector_dim; ++d) {
            cout << "mean[" << d << "]: " << *(mean + d) << endl;
            cout << "var[" << d << "]:" << *(var + d) << endl;
         }
         cout << "-----------------" << endl;
      }
   }
}

Cluster::~Cluster() {
}
