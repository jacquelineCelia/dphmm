#ifndef LOADER_H
#define LOADER_H

#include <string>
#include "cluster.h"
#include "data.h"
#include "model.h"

class Loader {
   public:
      Loader();
      Loader(const string, const string);
      void init(const string, const string);
      bool load_in_model(const int);
      bool load_in_data(const int);
      void load_out_fst(const string);
      void load_out_blabels(const string);
      void load_out_bmodels(const string);
      bool load_in_model(const Model&, const int);
      void save_labels(ofstream&);
      void save_train_token_number(ofstream&);
      void save_one_model(ofstream&, const int, const int);
      void save_comments(ofstream&, const string&);
      void save_one_mixture(ofstream&, const int, const int, const int);
      int get_num_clusters() {return cluster_num;}
      string convert_num_to_label(const int num);
      vector<Cluster*> get_models() {return clusters;}
      Data* get_data() {return data;}
      ~Loader();
   private:
      string fname;
      string fname_data;
      vector<Cluster*> clusters;
      Data* data;
      int cluster_num;
      int data_num; 
};

#endif
