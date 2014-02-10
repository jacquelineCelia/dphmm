#ifndef MIXTURE_H
#define MIXTURE_H

class Mixture {
   public:
      Mixture(const int);
      Mixture(const Mixture&);
      const Mixture& operator= (const Mixture&);
      void set_weight(const float);
      void set_det(const float);
      void set_mean(const float*);
      void set_var(const float*);
      int get_vector_dim() const {return vector_dim;}
      const float* get_mean() const {return mean;}
      const float* get_var() const {return var;}
      float get_det() const {return det;}
      float get_weight() const {return weight;}
      float compute_likelihood(const float*);
      ~Mixture();
   private:
      int vector_dim;
      float* mean;
      float* var;
      float weight;
      float det;
};

#endif
