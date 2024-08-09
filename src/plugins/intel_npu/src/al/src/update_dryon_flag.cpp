#pragma once
#include "update_dryon_flag.hpp"
#include <iostream>
#include <stdio.h>

namespace intel_npu {
bool DryonExecution::get_dryon_flag() {
    std::printf("   <>Current Dryon flag is %d\n", _flag_Dryon);
    return _flag_Dryon;
}

void DryonExecution::update_dryon_flag(bool flag) {
    std::printf("   <@>Updating Dryon flag from %d to %d\n", _flag_Dryon, flag);
    _flag_Dryon = flag;
}

}