#pragma once

namespace intel_npu {

class DryonExecution{
    bool get_dryon_flag();
    void update_dryon_flag(bool flag);
private:
    bool flag_Dryon = false;
};

extern DryonExecution globalDryonExecutionManager;

}