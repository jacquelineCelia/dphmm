#ifndef DATA_H
#define DATA_H

#include <vector>

using namespace std;

class Data {
   public:
      Data();
      Data(const int);
      void insert_frame(const float*);
      void set_frame_num();
      void set_dim(const int);
      void show();
      const float* get_frame_i(const int i) {return frames[i];}
      int get_dim() {return dim;}
      int get_frame_num() {return frame_num;}
      ~Data();
   private:
      vector<const float*> frames;
      int dim;
      int frame_num;
};

#endif
