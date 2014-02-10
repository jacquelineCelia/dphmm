#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <vector>

using namespace std;

class Calculator {
   public:
      Calculator();
      float sum_logs(vector<float>);
      float find_log_max(vector<float>);
      ~Calculator();
};

#endif
