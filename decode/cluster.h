#ifndef CLUSTER_H
#define CLUSTER_H

#include "gmm.h"

class Cluster {
   public:
      Cluster();
      void init(const int, const int, const int, const int);
      void set_prior(const float);
      void set_trans(const float*);
      void set_member_num(const int s_member_num) {member_num = s_member_num;}
      void set_state_mixture_weight(const int, const int, const float);
      void set_state_mixture_det(const int, const int, const float);
      void set_state_mixture_mean(const int, const int, const float*);
      void set_state_mixture_var(const int, const int, const float*);
      void set_state_tag(const int, const int);
      void set_state_trans(const int, const int, float);
      int get_state_num() const {return state_num;}
      int get_state_mixture_num(const int s) const {
         return states[s].get_mixture_num();} 
      int get_state_tag (const int) const;
      int get_id() const {return id;}
      int get_vector_dim() const {return vector_dim;}
      int get_member_num() const {return member_num;}
      float get_prior_prob() const {return prior;} 
      float get_trans_prob(const int i, const int j) const {return trans[i][j];}
      const float* get_state_mixture_mean(const int, const int);
      const float* get_state_mixture_var(const int, const int);
      float get_state_mixture_det(const int, const int);
      float get_state_mixture_weight(const int, const int);
      void init_prob_table(const int);
      float get_forward_prob(const int i, const int j) {return forward[i][j];}
      float get_backward_prob(const int i, const int j) {return backward[i][j];}
      void set_forward_prob(const int, const int, const float);
      void set_backward_prob(const int, const int, const float);
      float compute_state_prob(const int, const float*);
      void show_model();
      ~Cluster();
   private:
      vector<Gmm> states;
      vector<vector<float> > trans;
      vector<vector<float> > forward;
      vector<vector<float> > backward;
      int state_num;
      int mixture_num;
      int vector_dim;
      int id;
      int member_num;
      float prior;
};

#endif
