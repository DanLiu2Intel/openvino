#pragma once

namespace intel_npu {

class DryonExecution{
public:
    static DryonExecution& getDryonInstance() {  
        static DryonExecution instance;
        return instance;  
    }
    DryonExecution(const DryonExecution&) = delete;
    DryonExecution& operator=(const DryonExecution&) = delete;

    bool get_dryon_flag();
    void update_dryon_flag(bool flag);
private:
    bool _flag_Dryon;
    DryonExecution(): _flag_Dryon(false) {
        std::printf(" DryonExecution() flag = %d\n", _flag_Dryon);
    }
};

}