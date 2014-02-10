/* -*- C -*-
 *
 * Copyright (c) 1995, 2004
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 * 
 * Model.cpp
 *
 *  Created on: Mar 2, 2009
 *      Author: ydzhang
 */


#include "model.h"
#include "util.h"


Model::Model() {
  modelLoaded = false;
}

void Model::destroy() {

  modelLoaded = false;

  maxLabelLength = 0;

  SAFE_DEL_ARR(labels);
  SAFE_DEL_ARR(nTrainTokens);
  SAFE_DEL_ARR(logPrior);
  
  for(int i = 0; i < nModels; i++) {
    SAFE_DEL_ARR(nTrainTokensMix[i]);
    SAFE_DEL_ARR(logDet[i]);
    SAFE_DEL_ARR(logPriorMix[i]);
    for(int j = 0; j < nMixtures[i]; j++) {
      SAFE_DEL_ARR(mean[i][j]);
      SAFE_DEL_ARR(invCov[i][j]);
    }
    SAFE_DEL_ARR(mean[i]);
    SAFE_DEL_ARR(invCov[i]);
  }

  SAFE_DEL_ARR(nTrainTokensMix);

  SAFE_DEL_ARR(logDet);
  SAFE_DEL_ARR(logPriorMix);
  
  SAFE_DEL_ARR(mean);
  SAFE_DEL_ARR(invCov);

  SAFE_DEL_ARR(modelType);

}

Model::~Model() {

  destroy();

}

void Model::initNModels() {
  labels = new char * [nModels];
  nTrainTokens = new int [nModels];
  logPrior = new float [nModels];
  nMixtures = new int [nModels];

  nTrainTokensMix = new float * [nModels];
  mean = new float ** [nModels];
  invCov = new float ** [nModels];
  logDet = new float * [nModels];
  logPriorMix = new float * [nModels];

  maxLabelLength = 0;
}

void Model::initMixtures(int mIdx, int nMix) {
  mean[mIdx] = new float * [nMix];
  invCov[mIdx] = new float * [nMix];

  logDet[mIdx] = new float [nMix];
  logPriorMix[mIdx] = new float [nMix];

  nTrainTokensMix[mIdx] = new float [nMix];

  for(int i = 0; i < nMix; i++) {
    mean[mIdx][i] = new float [dim];
    invCov[mIdx][i] = new float [dim];
  }
}

void Model::loadMixture(ifstream & in, int mIdx, int mixIdx) {
  int oneInt;
  char oneChar;

  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);
  readScalar(in, oneChar, false);
  readScalar(in, oneInt, true);

  readScalar(in, nTrainTokensMix[mIdx][mixIdx], true);

  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);

  readVector(in, mean[mIdx][mixIdx], dim, true);

  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);

  readVector(in, invCov[mIdx][mixIdx], dim, true);

  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);

  if (oneInt != 2) {
    readScalar(in, logDet[mIdx][mixIdx], true);
    readScalar(in, logPriorMix[mIdx][mixIdx], true);
  } else {
    readScalar(in, logDet[mIdx][mixIdx], true);
    readScalar(in, oneInt, true);
    readScalar(in, logPriorMix[mIdx][mixIdx], true);
  }
  
}

void Model::loadOneModel(ifstream & in, int mIdx) {
  int oneInt;

  char * someChars = new char [maxLabelLength];

  readScalar(in, oneInt, true);

  readScalar(in, oneInt, true);
  readVector(in, someChars, oneInt, true);

  readScalar(in, nTrainTokens[mIdx], true);
  readScalar(in, logPrior[mIdx], true);
  readScalar(in, nMixtures[mIdx], true);

  initMixtures(mIdx, nMixtures[mIdx]);

  for(int i = 0; i < nMixtures[mIdx]; i++) {
    loadMixture(in, mIdx, i);    
  }

  SAFE_DEL_ARR(someChars);
}

void Model::loadLabels(ifstream & in) {
  int oneInt;
  for(int i = 0; i < nModels; i++) {
    readScalar(in, oneInt, true);
    readScalar(in, oneInt, true);
    labels[i] = new char [oneInt];
    readVector(in, labels[i], oneInt, true);
    (int)strlen(labels[i]) > maxLabelLength ? maxLabelLength = strlen(labels[i]) : 0;
  }
}

void Model::loadComments(ifstream & in) {
  char oneChar;
  in.read(reinterpret_cast<char *>(&oneChar), sizeof(char));
  while (oneChar != '\0') {
    in.read(reinterpret_cast<char *>(&oneChar), sizeof(char));
  }
}

int Model::load(const char * modelFile) {

  ifstream in(modelFile, ios_base::in | ios_base::binary);

  if (in.fail()) {
    cout << "Cannot open model file " << modelFile << endl;
    return 1;
  }

  // load begins

  int magicInt;

  readScalar(in, magicInt, true);

  if (magicInt != MODELS_V0) {
    cout << "Unsupported model format " << magicInt << endl;
    in.close();
    return 1;
  }

  int oneInt;

  readScalar(in, oneInt, true);

  modelType = new char [oneInt + 1];

  // model type

  readVector(in, modelType, oneInt + 1, false);

  // dimension and number of models

  readScalar(in, dim, true);
  readScalar(in, nModels, true);

  // format check
  readScalar(in, oneInt, true);

  readScalar(in, oneInt, true);

  if (oneInt != nModels) {
    cout << "mismatch " << oneInt << " : " << nModels << endl;
    in.close();
    return 1;
  }

  // init

  // allocate memory here
  // fatal errors occur from here must release memory before exit!

  initNModels();

  loadLabels(in);

  readScalar(in, oneInt, true);
  readScalar(in, oneInt, true);

  readVector(in, nTrainTokens, nModels, true);

  loadComments(in);

  readScalar(in, key, true);
  readScalar(in, randSeed, true);
  readScalar(in, expScale, true);

  readScalar(in, oneInt, true);

  for(int i = 0; i < nModels; i++) {
    loadOneModel(in, i);
  }

  modelLoaded = true;

  in.close();
  return 0;
}

void Model::writeModel(const char* fn_out, int repeat) {
   ofstream fout(fn_out, ios::binary);
   if (repeat > 1) {
      for(int i = 0; i < nModels; i++) {
         for(int j = 0; j < nMixtures[i]; j++) {
            logPriorMix[i][j] = log(exp(logPriorMix[i][j])/repeat);
         }
      }
   }
   for (int r = 0; r < repeat; ++r) {
      for(int i = 0; i < nModels; i++) {
         for(int j = 0; j < nMixtures[i]; j++) {
            fout.write(reinterpret_cast<char*> (&logPriorMix[i][j]), sizeof(float));
            cout << "Log prior = " << logPriorMix[i][j] << endl;
            fout.write(reinterpret_cast<char*> (mean[i][j]), sizeof(float) * dim);
            fout.write(reinterpret_cast<char*> (invCov[i][j]), sizeof(float) * dim);
            cout << "Mean Variance" << endl;
            for(int k = 0; k < dim; k++) {
               cout << "[" << k << "]=" << mean[i][j][k] << " [" << k << "]=" << invCov[i][j][k] << endl;
            }
            cout << "%%%%" << endl;
         }
      }
   }
   fout.close();
}

void Model::describe() {

  if (!modelLoaded) {
    cout << "Model not loaded" << endl;
    return;
  }

  cout << "*************************" << endl;
  cout << "N models = " << nModels << endl;
  cout << "Model type = " << modelType << endl;
  cout << "Key = " << key << endl;
  cout << "Random seed = " << randSeed << endl;
  cout << "Dimension = " << dim << endl;
  cout << "Exp scale = " << expScale << endl;
  cout << "*************************" << endl;

  for(int i = 0; i < nModels; i++) {
    cout << "-----------------------------" << endl;
    cout << "Model " << i << endl;
    cout << "N mixture = " << nMixtures[i] << endl;
    cout << "N train tokens = " << nTrainTokens[i] << endl;
    cout << "Log prior = " << logPrior[i] << endl;
    cout << "Label = " << labels[i] << endl;
    cout << "-----------" << endl;
    for(int j = 0; j < nMixtures[i]; j++) {
      cout << "Mixture " << j << endl;
      cout << "N train tokens = " << nTrainTokensMix[i][j] << endl;
      cout << "Log prior = " << logPriorMix[i][j] << endl;
      cout << "Log det = " << logDet[i][j] << endl;
      cout << "%%%%" << endl;
      cout << "Mean Variance" << endl;
      for(int k = 0; k < dim; k++) {
	cout << "[" << k << "]=" << mean[i][j][k] << " [" << k << "]=" << invCov[i][j][k] << endl;
      }
      cout << "%%%%" << endl;
    }
    cout << "-----------" << endl;
  }
}
