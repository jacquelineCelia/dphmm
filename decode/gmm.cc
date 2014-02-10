#include "gmm.h"

Gmm::Gmm(const int s_mixture_num, const int s_vector_dim) {
   mixture_num = s_mixture_num;
   vector_dim = s_vector_dim;
   for (int i = 0; i < mixture_num; ++i) {
      Mixture new_mixture(vector_dim);
      mix.push_back(new_mixture);
   }
}

void Gmm::set_mixture_weight(const int m, const float w) {
   mix[m].set_weight(w);
}

void Gmm::set_mixture_det(const int m, const float det) {
   mix[m].set_det(det);
}

void Gmm::set_mixture_mean(const int m, const float* mean) {
   mix[m].set_mean(mean);
}

const float* Gmm::get_mixture_mean(const int m) {
   return mix[m].get_mean();
}

void Gmm::set_mixture_var(const int m, const float* var) {
   mix[m].set_var(var);
}

const float* Gmm::get_mixture_var(const int m) {
   return mix[m].get_var();
}

float Gmm::get_mixture_det(const int m) {
   return mix[m].get_det();
}

float Gmm::get_mixture_weight(const int m) {
   return mix[m].get_weight();
}

float Gmm::compute_prob(const float* frame_i) {
   float log_probs_arr[mixture_num];
   for (int m = 0; m < mixture_num; ++m) {
      log_probs_arr[m] = mix[m].get_weight() + \
                         mix[m].compute_likelihood(frame_i);
   }
   vector<float> log_probs(log_probs_arr, \
     log_probs_arr + mixture_num);
   return calculator.sum_logs(log_probs); 
}

Gmm::~Gmm() {
}
