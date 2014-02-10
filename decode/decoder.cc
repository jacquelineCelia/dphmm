#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

#include "decoder.h"
#include "loader.h"
#include "calculator.h"
#include "model.h"

Decoder::Decoder(const string s_fn_model, \
  const string s_fn_data, const int s_dim) {
   fn_model = s_fn_model;
   fn_data = s_fn_data;
   dim = s_dim;
   loader.init(fn_model, fn_data);
}

bool Decoder::load_model(const int thre) {
   bool status = loader.load_in_model(thre);
   clusters = loader.get_models();
   return status;
}

bool Decoder::load_model(const Model& bmodels, const int thre) {
    bool status = loader.load_in_model(bmodels, thre);
    clusters = loader.get_models();
    return status;
}

bool Decoder::load_data() {
   bool status = loader.load_in_data(dim);
   data = loader.get_data();
   return status;
}

void Decoder::compute_posterior() {
   for (int i = 0; i < data -> get_frame_num(); ++i) {
      vector<float> frame_prob;
      frame_prob.clear();
      for (unsigned int c = 0; c < clusters.size(); ++c) {
         for (int s = 0; s < clusters[c] -> get_state_num(); ++s) {
            float frame_forward = clusters[c] -> get_forward_prob(s, i);
            float frame_backward = clusters[c] -> get_backward_prob(s, i);
            frame_prob.push_back(frame_forward + frame_backward);
         }
      }
      float normalizer = calculator.sum_logs(frame_prob);
      for (unsigned int l = 0; l < frame_prob.size(); ++l) {
         frame_prob[l] -= normalizer;
      }
      posteriors.push_back(frame_prob);
   }
}

void Decoder::compute_forward_backward() {
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      clusters[c] -> init_prob_table(data -> get_frame_num());
   }
   compute_forward_prob();
   compute_backward_prob();
}

void Decoder::compute_single_state() {
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      clusters[c] -> init_prob_table(data -> get_frame_num());
   }
   for (int i = 0; i < data -> get_frame_num(); ++i) {
      const float* frame_i = data -> get_frame_i(i);
      for (unsigned int c = 0; c < clusters.size(); ++c) {
         for (int s = 0; s < clusters[c] -> get_state_num(); ++s) {
            float likelihood = clusters[c] -> compute_state_prob( \
              s, frame_i); 
            clusters[c] -> set_forward_prob(s, i, likelihood);
         }
      }
   }
}

void Decoder::compute_forward_prob() {
   // for the first frame, code directly
   // after the first frame, use the general pattern
   const float* frame_i = data -> get_frame_i(0);
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      float likelihood = clusters[c] -> compute_state_prob(0, frame_i);
      float prior = clusters[c] -> get_prior_prob();
      clusters[c] -> set_forward_prob(0, 0, likelihood + prior);
      for (int s = 1; s < clusters[c] -> get_state_num(); ++s) {
         clusters[c] -> set_forward_prob(s, 0, log(0));
      }
   }
   for (int i = 1; i < data -> get_frame_num(); ++i) {
      for (unsigned int c = 0; c < clusters.size(); ++c) {
         for (int s = 0; s < clusters[c] -> get_state_num(); ++s) {
            // Compute the forward prob for each state of each cluster
            if (s == 0) {
               vector<float> income;
               for (unsigned int pre = 0; pre < clusters.size(); ++pre) {
                  int last_state = clusters[pre] -> get_state_num() - 1; 
                  income.push_back(\
                     // clusters[pre] -> get_forward_prob(last_state, i - 1));
                     clusters[pre] -> get_forward_prob(last_state, i - 1) + \
                     clusters[pre] -> get_trans_prob(last_state, last_state + 1));
                  income.push_back(\
                    // clusters[pre] -> get_forward_prob(0, i - 1));
                     clusters[pre] -> get_forward_prob(0, i - 1) + \
                     clusters[pre] -> get_trans_prob(0, last_state + 1));
               }
               float inter_prob = calculator.sum_logs(income);
               inter_prob += clusters[c] -> get_prior_prob();
               float intra_prob = clusters[c] -> get_forward_prob(0, i - 1) + \
                                  clusters[c] -> get_trans_prob(0, 0);
               income.clear();
               income.push_back(inter_prob);
               income.push_back(intra_prob);
               float trans_prob = calculator.sum_logs(income);
               float likelihood = clusters[c] -> compute_state_prob(\
                 0, data -> get_frame_i(i));
               clusters[c] -> set_forward_prob(0, i, trans_prob + likelihood);
            }
            else {
               vector<float> income;
               for (int pre_state = 0; pre_state <= s; ++pre_state) {
                  income.push_back(\
                    clusters[c] -> get_forward_prob(pre_state, i - 1) + \
                    clusters[c] -> get_trans_prob(pre_state, s));
               }
               float trans_prob = calculator.sum_logs(income);
               float likelihood = clusters[c] -> compute_state_prob(\
                 s, data -> get_frame_i(i));
               clusters[c] -> set_forward_prob(s, i, trans_prob + likelihood);
            }
         }
      }
   }
}

void Decoder::compute_backward_prob() {
   int last_frame_index = data -> get_frame_num() - 1;
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      for (int s = 0; s < clusters[c] -> get_state_num(); ++s) {
         clusters[c] -> set_backward_prob(s, last_frame_index, 0);
      }
   }
   int i = data -> get_frame_num() - 2;
   for (; i >= 0; --i) {
      for (unsigned int c = 0; c < clusters.size(); ++c) {
         for (int s = 0; s < clusters[c] -> get_state_num(); ++s) {
            vector<float> outcome;
            for (int next_state = s; \
              next_state < clusters[c] -> get_state_num(); ++next_state) {
               outcome.push_back( \
                 clusters[c] -> get_backward_prob(next_state, i + 1) + \
                 clusters[c] -> compute_state_prob(next_state, \
                    data -> get_frame_i(i + 1)) + \
                 clusters[c] -> get_trans_prob(s, next_state));
            }
            float intra_prob = calculator.sum_logs(outcome);
            if (s == 0 || s == clusters[c] -> get_state_num() - 1) {
               vector<float> outcome;
               for (unsigned int next = 0; next < clusters.size(); ++next) {
                  outcome.push_back(\
                    clusters[next] -> get_backward_prob(0, i + 1) + \
                    clusters[next] -> compute_state_prob(\
                       0, data -> get_frame_i(i + 1)) + \
                    clusters[next] -> get_prior_prob());
               }
               float inter_prob = calculator.sum_logs(outcome);
               inter_prob += clusters[c] -> get_trans_prob(\
                  s, clusters[c] -> get_state_num());
               clusters[c] -> set_backward_prob(s, i, inter_prob + intra_prob);
            }
            else {
               clusters[c] -> set_backward_prob(s, i, intra_prob);
            }
         }
      }
   }
}

void Decoder::save_posterior(const string fn_out) {
   ofstream fout(fn_out.c_str(), ios::out);
   for (int i = 0 ; i < data -> get_frame_num(); ++i) {
      stringstream frame_index; 
      frame_index << i;
      string output = frame_index.str();
      for (unsigned int l = 0; l < posteriors[i].size(); ++l) {
         stringstream feature_index;
         feature_index << l; 
         stringstream feature_value;
         feature_value << posteriors[i][l];
         output += " " + \
            feature_index.str() + ":" + feature_value.str(); 
      }
      output += "\n";
      fout << output;
   }
   fout.close();
}

void Decoder::save_posterior_bin(const string fn_out) {
   ofstream fout(fn_out.c_str(), ios::out | ios::binary);
   for (int i = 0 ; i < data -> get_frame_num(); ++i) {
      for (unsigned int l = 0; l < posteriors[i].size(); ++l) {
          fout.write(reinterpret_cast<char*> (&posteriors[i][l]), sizeof(float));
      }
   }
   fout.close();
}

bool Decoder::load_hdphmm_model(const int threshold) {
    return false;
}

Decoder::~Decoder() {
}
