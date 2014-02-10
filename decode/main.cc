#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "decoder.h"
#include "model.h"

void print_usage() {
   cout << "./decode_to_pg -i model_file -d data_file"
     << " -o output_file -d dim -m mode[1: forward backward"
     << " 2: single state decoding] -t threshod"
     << " -a model_type[0: dphmm, 1: bmodel, 2:hdphmm] "
     << " -b output in binary form[1:yes, 0:no]" << endl;
}

int main(int argc, char* argv[]) {
   if (argc != 17) {
      print_usage();
      return 1;
   }
   string fname_model = argv[2];
   string fname_data = argv[4];
   string fname_out = argv[6];
   int dim = atoi(argv[8]);
   int mode = atoi(argv[10]);
   int threshold = atoi(argv[12]);
   int from_bmodels = atoi(argv[14]);
   int binary_out = atoi(argv[16]);

   Decoder decoder(fname_model, fname_data, dim);

   if (from_bmodels == 1) {
      Model bmodels;
      bmodels.load(fname_model.c_str());
      if (!decoder.load_model(bmodels, threshold)) {
           cout << "Model file cannot be opened. " 
                << "Check " << fname_model << "..." << endl;
           return -1;
      }
      else {
        cout << "model loaded" << endl;
      }
   }
   else if (from_bmodels == 0){
       if (!decoder.load_model(threshold)) {
           cout << "Model file cannot be opened. " 
                << "Check " << fname_model << "..." << endl;
           return -1;
       }
       else {
           cout << "model loaded" << endl;
       }
   }
   else {
       if (!decoder.load_hdphmm_model(threshold)) {
           cout << "Model file cannot be opened. " 
                << "Check " << fname_model << "..." << endl;
           return -1;
       }
       else {
           cout << "model loaded" << endl;
       }
   }
   if (!decoder.load_data()) {
      cout << "Data file cannot be opened. "
        << "Check " << fname_data << "..." << endl;
      return -1;
   }
   else {
      cout << "data loaded" << endl;
   }
   if (mode == 1) {
      decoder.compute_forward_backward();
      decoder.compute_posterior();
   }
   else if (mode == 2) {
      decoder.compute_single_state();
      cout << "single state computed" << endl;
      decoder.compute_posterior();
      cout << "pg computed" << endl;
   }
   cout << "ready to save" << endl;
   if (binary_out) {
       decoder.save_posterior_bin(fname_out);
   }
   else {
       decoder.save_posterior(fname_out);
   }
   cout << "saved" << endl;
   return 0;
}
