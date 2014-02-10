/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

* ./load_bmodels.cc
*	FILE: cluster.cc 				                                *
*										                            *
*   				      				                            *
*   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>					*
*   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "model.h"
#include "loader.h"

void print_usage() {
    cout << "load_bmodel -in .bmodel" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage();
        return 1;
    }
    Model bmodels;
    Loader loader;
    string fname_bmodel = argv[2];
    bmodels.load(fname_bmodel.c_str()); 
    bmodels.describe();
    loader.load_in_model(bmodels, 0);
    return 0;
}
