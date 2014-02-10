/* -*- C -*-
 *
 * Copyright (c) 1995, 2004
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 * 
 * Model.h
 *
 *  Created on: Mar 2, 2009
 *      Author: ydzhang
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <fstream>
#include <iostream>
#include <string.h>

using namespace std;

class Model {
 public:
  
  Model();
  ~Model();

  int load(const char * modelFile);
  void describe();
  void destroy();

  bool modelLoaded;

  char * modelType;

  int key;
  int randSeed;

  int dim;

  float expScale;

  int maxLabelLength;

  int nModels;
  int * nMixtures;
  int * nTrainTokens;
  float * logPrior;
  char ** labels;

  float ** nTrainTokensMix;
  float ** logPriorMix;  
  float ** logDet;

  float *** invCov;
  float *** mean;
  void writeModel(const char*, int);
 private:

  void initNModels();
  void initMixtures(int mIdx, int nMix);
  void loadOneModel(ifstream & in, int mIdx);
  void loadLabels(ifstream & in);
  void loadComments(ifstream & in);
  void loadMixture(ifstream & in, int mIdx, int mixIdx);
};

#endif /* MODEL_H_ */
