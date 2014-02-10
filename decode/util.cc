/* -*- C -*-
 *
 * Copyright (c) 1995, 2004
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 * */


#include "util.h"


void stringTokenize(const string & str, vector<string> & tokens, string & delimiters) {
  string::size_type last_pos = str.find_first_not_of(delimiters, 0);
  string::size_type pos = str.find_first_of(delimiters, last_pos);
  while (string::npos != pos || string::npos != last_pos) {
    tokens.push_back(str.substr(last_pos, pos - last_pos));
    last_pos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, last_pos);
  }
}
