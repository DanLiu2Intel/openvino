#pragma once


#include <iostream>
#include "update_dryon_flag.hpp"
#include "/home/dl5w0502/vpux/openvino/src/plugins/intel_npu/src/plugin/include/plugin.hpp" 

namespace intel_npu {
bool flag_Dryon = false;

void update_dryon_flag(bool flag) {
    flag_Dryon = true;
    if(flag_Dryon)
        std::cout << " !!!! true"<< std::endl;
    else
        std::cout << " !!!! flase"<< std::endl;
}

}