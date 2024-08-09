#pragma once
#include "update_dryon_flag.hpp"
#include <iostream>
#include <stdio.h>

namespace intel_npu {
bool DryonExecution::get_dryon_flag() {
    if(_flag_Dryon)
        std::cout << " (get)!!!! true"<< std::endl;
    else
        std::cout << " (get)!!!! flase"<< std::endl;
    return _flag_Dryon;
}

void DryonExecution::update_dryon_flag(bool flag) {
    std::printf("     --------before update, %d\n", get_dryon_flag());
    if(_flag_Dryon)
        std::cout << " !!!! true"<< std::endl;
    else
        std::cout << " !!!! flase"<< std::endl;


    _flag_Dryon = true;
    if(_flag_Dryon)
        std::cout << " !!!! true"<< std::endl;
    else
        std::cout << " !!!! flase"<< std::endl;
    
    std::printf("     --------after update, %d\n", get_dryon_flag());
}

}