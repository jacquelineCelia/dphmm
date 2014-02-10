/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

* ./loader.cc
*	FILE: cluster.cc 				                                *
*										                            *
*   				      				                            *
*   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>					*
*   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>

#include "loader.h"
#include "cluster.h"
#include "util.h"
#include "model.h"

#define LTPI 1.8378770664093454835606594728112 

using namespace std;

Loader::Loader() {
}

Loader::Loader(const string s_filename, \
  const string s_filename_data) {
   fname = s_filename;
   fname_data = s_filename_data;
}

void Loader::init(const string s_filename, \
  const string s_filename_data) {
   fname = s_filename;
   fname_data = s_filename_data;
}


bool Loader::load_in_model(const int threshold) {
   ifstream fin(fname.c_str(), ios::binary);
   if (!fin.good()) {
      cout << fname << " cannot be opened." << endl;
      return false;
   }
   fin.read(reinterpret_cast<char*> (&data_num), sizeof(int));
   fin.read(reinterpret_cast<char*> (&cluster_num), sizeof(int));
   for (int i = 0; i < cluster_num; ++i) {
      Cluster* new_cluster = new Cluster;
      int member_num;
      int state_num;
      int mixture_num;
      int vector_dim;
      fin.read(reinterpret_cast<char*> (&member_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&state_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&mixture_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&vector_dim), sizeof(int));
      new_cluster -> init(state_num, mixture_num, vector_dim, i);
      new_cluster -> set_prior(log((float) member_num / data_num));
      new_cluster -> set_member_num(member_num);
      float trans[state_num * (state_num + 1)];
      fin.read(reinterpret_cast<char*> (trans), sizeof(float) * \
        state_num * (state_num + 1));
      new_cluster -> set_trans(trans);
      for (int j = 0; j < state_num; ++j) {
         new_cluster -> set_state_tag(j, i * (state_num) + j);
         for (int k = 0 ; k < mixture_num; ++k) {
            float w;
            float det;
            float mean[vector_dim];
            float var[vector_dim];
            fin.read(reinterpret_cast<char*> (&w), \
              sizeof(float));
            fin.read(reinterpret_cast<char*> (&det), \
              sizeof(float));
            fin.read(reinterpret_cast<char*> (mean), \
              sizeof(float) * vector_dim);
            fin.read(reinterpret_cast<char*> (var), \
              sizeof(float) * vector_dim);
            new_cluster -> set_state_mixture_weight(j, k, w);
            new_cluster -> set_state_mixture_det(j, k, det);
            new_cluster -> set_state_mixture_mean(j, k, mean);
            new_cluster -> set_state_mixture_var(j, k, var);
         }
      }
      if (new_cluster -> get_member_num() > threshold) {
         clusters.push_back(new_cluster);
      }
      else {
         data_num -= new_cluster -> get_member_num();
         delete new_cluster;
      }
   }
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      float member_num = (float) clusters[c] -> get_member_num();
      clusters[c] -> set_prior(log(member_num / (float) data_num));
      clusters[c] -> show_model();
   }
   fin.close();
   return true;
}

bool Loader::load_in_data(const int dim) {
   data = new Data(dim); 
   ifstream fdata(fname_data.c_str(), ios::binary);
   if (!fdata.good()) {
      cout << fname_data << " cannot be opened." << endl;
      return false;
   }
   fdata.seekg(0, ios_base::end);
   int data_len = (int) fdata.tellg();
   fdata.seekg(0, ios_base::beg);
   int counter = 0;
   cout << "loading data..." << endl;
   while (fdata.tellg() < data_len) {
      float* frame_i = new float[dim];
      fdata.read(reinterpret_cast<char*> (frame_i), sizeof(float) * dim);
      data -> insert_frame(frame_i);
      counter++;
   }
   data -> set_frame_num();
//   data -> show();
   fdata.close();
   return true;
}

string Loader::convert_num_to_label(const int s_num) {
   string label;
   int num = s_num;
   int d = num % 26 + 97;
   char t = d;
   label += t; 
   num /= 26;
   while (num > 0) {
      d = num % 26 + 97;
      t = d;
      label += t; 
      num /= 26;
   }
   return label;
}

void Loader::load_out_fst(const string fname_fst) {
   ofstream fst(fname_fst.c_str(), ios::out);
   fst << "I 0" << endl;
   int total_states = 0;
   for (int i = 0; i < cluster_num; ++i) {
      total_states += clusters[i] -> get_state_num();
      int start_state_tag = clusters[i] -> get_state_tag(0); 
      fst << "T 0 " << start_state_tag + 1 << " t." << start_state_tag 
          << " " << convert_num_to_label(clusters[i] -> get_id()) << " " 
          << -(clusters[i] -> get_prior_prob()) << endl;
      int state_num = clusters[i] -> get_state_num();
      for (int j = 0; j < state_num; ++j) {
         int start_tag = clusters[i] -> get_state_tag(j);
         if (j == 1) {
            float saved_prob = 0;
            for (int k = j; k < state_num; ++k) {
               int end_tag = clusters[i] -> get_state_tag(k);
               float prob = exp(clusters[i] -> get_trans_prob(j, k));
               if (prob > 0.05) {
                  prob -= 0.05; 
                  saved_prob += 0.05;
               }
               fst << "T " << start_tag + 1 << " " << end_tag + 1
                   << " i." << end_tag << " , " 
                  << -(log(prob)) << endl;
            }
            clusters[i] -> set_state_trans(j, state_num, log(saved_prob));
         }
         else {
            for (int k = j; k < state_num; ++k) {
               int end_tag = clusters[i] -> get_state_tag(k);
               fst << "T " << start_tag + 1 << " " << end_tag + 1
                   << " i." << end_tag << " , " 
                  << -(clusters[i] -> get_trans_prob(j, k)) << endl;
            }
         }
      }
   }
   ++total_states;
   for (int i = 0; i < cluster_num; ++i) {
      int start_state_tag = clusters[i] -> get_state_tag(0);
      int state_num = clusters[i] -> get_state_num();
      fst << "T " << start_state_tag + 1 << " " << total_states << " , , "
        << -(clusters[i] -> get_trans_prob(0, state_num)) << endl;
      int middle_state_tag = clusters[i] -> get_state_tag(1);
      fst << "T " << middle_state_tag + 1 << " " << total_states << " , , "
        << -(clusters[i] -> get_trans_prob(1, state_num)) << endl;
      int last_state_tag = clusters[i] -> get_state_tag(state_num - 1); 
      fst << "T " << last_state_tag  + 1 << " " << total_states << " , , "
        << -(clusters[i] -> get_trans_prob(state_num - 1, state_num)) << endl;
   }
   fst << "T " << total_states << " 0" << endl;
   fst << "F " << total_states << endl;
   fst.close();
}

void Loader::load_out_blabels(const string fname_blabels) {
   ofstream flabels(fname_blabels.c_str(), ios::out);
   for (int i = 0 ; i < cluster_num; ++i) {
      int state_num = clusters[i] -> get_state_num();
      for (int j = 0; j < state_num; ++j) {
         flabels << "m{" << clusters[i] -> get_state_tag(j) << "}" << endl;
      }
   }
   flabels.close();
}

void Loader::load_out_bmodels(const string fname_bmodels) {
   ofstream fmodels(fname_bmodels.c_str(), ios::binary);
   int model_num = cluster_num * clusters[0] -> get_state_num(); 
   // Write magicInt defined in Summit
   int magicInt = MODELS_V0;
   writeScalar(fmodels, magicInt, true);
   char modelType[26] = "mixture_diagonal_gaussian";
   int modelType_len = strlen(modelType); 
   writeScalar(fmodels, modelType_len, true);
   writeVector(fmodels, modelType, modelType_len + 1, true);
   // Write dimension and number of models
   int vectorDim = clusters[0] -> get_vector_dim();
   writeScalar(fmodels, vectorDim, true);
   writeScalar(fmodels, model_num, true);
   int oneInt = 111;
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, model_num, true);
   save_labels(fmodels);
   oneInt = 14;
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, model_num, true);
   save_train_token_number(fmodels);
   string comments = "";
   save_comments(fmodels, comments);
   oneInt = 1;
   float expScale = 1.0;
   // Write key, randSeed, expScale
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, expScale, true);
   writeScalar(fmodels, model_num, true);
   // Write models
   for(int i = 0; i < cluster_num; ++i) {
      for (int j = 0; j < clusters[i] -> get_state_num(); ++j) {
         save_one_model(fmodels, i, j);
      }
   }
   fmodels.close();
}

void Loader::save_one_model(ofstream& fmodels, const int c, const int s) {
   int oneInt;
   oneInt = 11;
   writeScalar(fmodels, oneInt, true);
   // make the label
   stringstream num_to_string;
   num_to_string << clusters[c] -> get_state_tag(s);
   string label = "m{" + num_to_string.str() + "}";
   char label_char[100];
   strcpy(label_char, label.c_str());
   int label_len = label.length() + 1;
   writeScalar(fmodels, label_len, true);
   writeVector(fmodels, label_char, label_len, true);
   // Write number of training tokens
   int trainTokens = 100; 
   writeScalar(fmodels, trainTokens, true); 
   float logPrior = 0; 
   writeScalar(fmodels, logPrior, true);
   int mixture_num = clusters[c] -> get_state_mixture_num(s);
   writeScalar(fmodels, mixture_num, true);
   for (int m = 0; m < mixture_num; ++m) {
      save_one_mixture(fmodels, c, s, m);
   }
}

void Loader::save_one_mixture(ofstream& fmodels, const int c, \
  const int s, const int m) {
   int oneInt = 11;
   writeScalar(fmodels, oneInt, true);
   oneInt = 1;
   writeScalar(fmodels, oneInt, true);
   char oneChar = '\0';
   fmodels.write(reinterpret_cast<char*> (&oneChar), sizeof(char));
   int dim = clusters[c] -> get_vector_dim();
   writeScalar(fmodels, dim, true);
   oneInt = 100;
   writeScalar(fmodels, oneInt, true);
   oneInt = 14;
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, dim, true);
   float mean[dim];
   memcpy(mean, clusters[c] -> get_state_mixture_mean(s, m), sizeof(float) * dim);
   writeVector(fmodels, mean, dim, true);
   oneInt = 1;
   writeScalar(fmodels, oneInt, true);
   oneInt = 14;
   writeScalar(fmodels, oneInt, true);
   writeScalar(fmodels, dim, true);
   float var[dim];
   memcpy(var, clusters[c] -> get_state_mixture_var(s, m), sizeof(float) * dim);
   writeVector(fmodels, var, dim, true);
   oneInt = 0;
   writeScalar(fmodels, oneInt, true);
   oneInt = 14;
   writeScalar(fmodels, oneInt, true);
   oneInt = 2;
   writeScalar(fmodels, oneInt, true);
   float det = clusters[c] -> get_state_mixture_det(s, m);
   writeScalar(fmodels, det, true);
   oneInt = 0;
   writeScalar(fmodels, oneInt, true);
   float weight = clusters[c] -> get_state_mixture_weight(s, m);
   writeScalar(fmodels, weight, true);
}

void Loader::save_comments(ofstream& fmodels, const string& comments) {
   for (unsigned int i = 0; i < comments.length(); ++i) {
      char t = comments[i];
      fmodels.write(reinterpret_cast<char*> (&t), sizeof(char));
   }
   char t = '\0';
   fmodels.write(reinterpret_cast<char*> (&t), sizeof(char));
}

void Loader::save_train_token_number(ofstream& fmodels) {
   int model_num = cluster_num * clusters[0] -> get_state_num();
   int trainTokens[model_num];
   for (int i = 0; i < model_num; ++i) {
      trainTokens[i] = 100;
   }
   writeVector(fmodels, trainTokens, model_num, true);
}

void Loader::save_labels(ofstream& fmodels) {
   int oneInt;
   for (int i = 0 ; i < cluster_num; ++i) {
      for (int j = 0; j < clusters[i] -> get_state_num(); ++j) {
         oneInt = 11;
         writeScalar(fmodels, oneInt, true);
         stringstream num_to_string;
         num_to_string << clusters[i] -> get_state_tag(j);
         string label = "m{" + num_to_string.str() + "}";
         char label_char[200];
         strcpy(label_char, label.c_str());
         int label_len = label.length() + 1;
         writeScalar(fmodels, label_len, true);
         writeVector(fmodels, label_char, label_len, true);
      }
   }
}

bool Loader::load_in_model(const Model& bmodels, const int threshold) {
   int data_num = 0;
   int cluster_num = bmodels.nModels;
   for (int i = 0; i < cluster_num; ++i) {
      data_num += bmodels.nTrainTokens[i];
   }
   for (int i = 0; i < cluster_num; ++i) {
      Cluster* new_cluster = new Cluster;
      int member_num  = bmodels.nTrainTokens[i];
      int state_num = 1; 
      int mixture_num = bmodels.nMixtures[i];
      int vector_dim = bmodels.dim;
      new_cluster -> init(state_num, mixture_num, vector_dim, i);
      new_cluster -> set_prior(log((float) member_num / data_num));
      new_cluster -> set_member_num(member_num);
      float trans[state_num * (state_num + 1)];
      for (int j = 0; j < state_num; ++j) {
        for (int k = 0; k < state_num + 1; ++k) {
            trans[(state_num + 1)* j + k] = 0;
        }
      }
      new_cluster -> set_trans(trans);
      for (int j = 0; j < state_num; ++j) {
        new_cluster -> set_state_tag(j, i * (state_num) + j);
        for (int k = 0; k < mixture_num; ++k) {
            float w;
            float det;
            float mean[vector_dim];
            float var[vector_dim];
            w = bmodels.logPriorMix[i][k];
            det = bmodels.logDet[i][k];
            det += vector_dim * LTPI;
            det *= -0.5;
            memcpy(mean, bmodels.mean[i][k], \
                    sizeof(float) * vector_dim);
            memcpy(var, bmodels.invCov[i][k], \
                    sizeof(float) * vector_dim);
            new_cluster -> set_state_mixture_weight(j, k, w);
            new_cluster -> set_state_mixture_det(j, k, det);
            new_cluster -> set_state_mixture_mean(j, k, mean);
            new_cluster -> set_state_mixture_var(j, k, var);
        }
      }
      if (new_cluster -> get_member_num() > threshold) {
        clusters.push_back(new_cluster);
      }
      else {
         data_num -= new_cluster -> get_member_num();
         delete new_cluster;
      }
   }
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      float member_num = (float) clusters[c] -> get_member_num();
      clusters[c] -> set_prior(log(member_num / (float) data_num));
//      clusters[c] -> show_model();
   }
   return true;
} 

Loader::~Loader() {
   // delete clusters
   for (unsigned int c = 0; c < clusters.size(); ++c) {
      delete clusters[c];
   }
}
