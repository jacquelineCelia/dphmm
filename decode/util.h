/* -*- C -*-
 *
 * Copyright (c) 1995, 2004
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 * 
 * Util.h
 *
 *  Created on: Mar 2, 2009
 *      Author: ydzhang
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

#include <math.h>
#include <string.h>

// pointer operations

#define SAFE_DEL(p) {if (p) { delete (p); (p) = NULL; } }
#define SAFE_DEL_ARR(p) {if (p) { delete [] (p); (p) = NULL; } }

// math constants

#define PI   3.1415926535897932384626433832795
#define TPI  6.2831853071795864769252867665590		// PI*2
#define LTPI 1.8378770664093454835606594728112		// log(PI*2)
#define EXP  2.7182818284590452353602874713527

#define LZERO  -1.0E10		// ~log(0)
#define LSMALL -0.5E10		// log values < LSMALL are set to LZERO
#define THRESH 1E-6f

#define MINEARG -708.3f		// lowest exp() arg  = log(MINLARG)
#define MINLARG 2.45E-308f	// lowest log() arg  = exp(MINEARG)

#define LN10 2.3025850929940456840179914546844
#define LN2  0.69314718055994530941723212145818
#define LN3 1.0986122886681096913952452369225

#define XACC 0.0000001 //accurace of the solution

#define GAMMA_FLOOR 1.0E-10f

// feature format

#define TRACK_V0 0x56782
#define FLAT_MFCC 101
#define CLUSTER_MFCC 16777216

// model format

#define MODELS_V0 422160

using namespace std;

// string operations

void stringTokenize(const string & str, vector<string> & tokens, string & delimiters);

// byte operations

template <class T> T byteReverse(T val) {
  unsigned char* p = reinterpret_cast<unsigned char*>(&val);
  reverse(p, p + sizeof(val));
  return val;
}

template <class T> void vectorReverse(T * ptr, int length) {
  for(int i = 0; i < length; i++) {
    ptr[i] = byteReverse(ptr[i]);
  }
}

// IO operations

template <class T> void readScalar(ifstream & in, T & val, bool rev) {
  in.read(reinterpret_cast<char *>(&val), sizeof(T));
  val = rev ? byteReverse(val) : val;
}

template <class T> void readVector(ifstream & in, T * ptr, int length, bool rev) {
  in.read(reinterpret_cast<char *>(ptr), sizeof(T) * length);
  if (rev) {
    vectorReverse(ptr, length);
  }
}

template <class T> void writeScalar(ofstream & out, T & val, bool rev) {
  T tmp = rev ? byteReverse(val) : val;
  out.write(reinterpret_cast<const char *>(&tmp), sizeof(T));
}

template <class T> void writeVector(ofstream & out, T * ptr, int length, bool rev) {
  if (rev) {
    T * tmp = new T[length];
    memcpy(tmp, ptr, sizeof(T) * length);
    vectorReverse(tmp, length);
    out.write(reinterpret_cast<const char *>(tmp), sizeof(T) * length);
    SAFE_DEL_ARR(tmp);
  } else {
    out.write(reinterpret_cast<const char *>(ptr), sizeof(T) * length);
  }
}

// math operations

template <class T> T floorExp(T x) {
  return (x < MINEARG) ? 0.0 : exp(x);
}

template <class T> T logAdd(T x, T y) {
  T tmp, diff, z;
  if (x < y) {
    tmp = x;
    x = y;
    y = tmp;
  }
  diff = y - x;
  if (diff < -log(-LZERO))
    return (x < LSMALL) ? LZERO : x;
  else {
    z = floorExp(diff);
    return x + log(1.0 + z);
  }
}

template <class T> T logSumExp(T * x, int size) {
  T maxVal = LSMALL;

  for(int i = 0; i < size; i++) {
    if (x[i] > maxVal) {
      maxVal = x[i];
    }
  }

  T sumExp = 0.0;

  for(int i = 0; i < size; i++) {
    sumExp += floorExp(x[i] - maxVal);
  }

  return (log(sumExp) + maxVal);
}

/*template <class T> T logSub(T x, T y) {
  T diff, z;

  if (x < y) {
    return LZERO;
  }

  diff = y - x;

  if (diff < -log(-LZERO)) {
    return (x < LSMALL) ? LZERO : x;
  } else {
    z = 1.0 - floorExp(diff);
    return (z < MINLARG) ? LZERO : x + log(z);
  }
  }*/

#endif
