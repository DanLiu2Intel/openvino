// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type.hpp"

#include "openvino/util/common_util.hpp"
#include <cstring>
#include <iostream>
#include <cstdlib>
namespace std {
size_t std::hash<ov::DiscreteTypeInfo>::operator()(const ov::DiscreteTypeInfo& k) const {
    return k.hash();
}
}  // namespace std

namespace ov {

size_t DiscreteTypeInfo::hash() const {
    if (hash_value != 0) {
        std::cout << "DiscreteTypeInfo::hash() is called, hash_value: " << hash_value << std::endl;
        return hash_value;
    }
    size_t name_hash = name ? std::hash<std::string>()(std::string(name)) : 0;
    size_t version_id_hash = version_id ? std::hash<std::string>()(std::string(version_id)) : 0;

    return ov::util::hash_combine(std::vector<size_t>{name_hash, version_id_hash});
}

size_t DiscreteTypeInfo::hash() {
    if (hash_value == 0)
        hash_value = static_cast<const DiscreteTypeInfo*>(this)->hash();
    return hash_value;
}

bool DiscreteTypeInfo::is_castable(const DiscreteTypeInfo& target_type) const {
    return *this == target_type || (parent && parent->is_castable(target_type));
}

std::string DiscreteTypeInfo::get_version() const {
    if (version_id) {
        return std::string(version_id);
    }
    return {};
}

DiscreteTypeInfo::operator std::string() const {
    std::printf("7nd=========check point1============\n");///
    if(name == nullptr) {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, name is nullptr" << std::endl;  
    }/// no
    for (int i = 0; i < 2; i++) {
        std::printf("+%c", name[i]);
    }
    std::printf("\n");
    std::printf("===operator std::string()1===\n");
    char* env = std::getenv("PRINT");
    std::printf("===operator std::string()2===\n");
    if (env != nullptr) {
        std::cout << "++ENV is: " << env << std::endl;
        if (std::string(env) == "P6") {
            std::cout << "++ENV is: P6        " << env << std::endl;
            for (int i = 0; i < 6; i++) {
                std::printf("+6+%c", name[i]);
            }
            std::printf("\n");
        } else if (std::string(env) == "P10") {
            std::cout << "++ENV is: P10    " << env << std::endl;
            for (int i = 0; i < 10; i++) {
                std::printf("+10+%c", name[i]);
            }
            std::printf("\n");
        } else if (std::string(env) == "P15") {
            std::cout << "++ENV is: P15    " << env << std::endl;
            for (int i = 0; i < 15; i++) {
                std::printf("+15+%c", name[i]);
            }
            std::printf("\n");
        } else if (std::string(env) == "PALL") {
                std::cout << "++ENV is: PALL    " << env << std::endl;
            for (int i = 0; name[i] != '\0'; i++) {
                std::printf("++%c", name[i]);
            }
            std::printf("\n");
        }
    } else {
        std::cout << "++ENV is nullptr" << std::endl;
    }

    std::printf("==========check point2===========\n");////
    if(version_id == nullptr) {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, version_id is nullptr" << std::endl;////
    }

    std::printf("==========check point3===========\n");
    if(version_id == nullptr) {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 22version_id is nullptr" << std::endl;////
    } else {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 22get_version: " << get_version() << std::endl;
    }
    std::printf("==========check point4===========\n");////
    if(name == nullptr) {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 22name is nullptr" << std::endl;
    } else {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 22name: " << name << std::endl;
    }
    std::printf("==========check point5===========\n");
    if(get_version().empty()) {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 33version_id is nullptr" << std::endl;
    } else {
        std::cout << "DiscreteTypeInfo::operator std::string() is called, 33get_version: " << get_version() << std::endl;
    }
    std::printf("==========check point6===========\n");
    std::string version_str = get_version().empty() ? "(empty)" : get_version();
    std::printf("==========check point7===========version_str is %s\n", version_str.c_str());
    std::string nam_str = name ? std::string(name) : "(empty)";
    std::printf("==========check point8===========nam_str is %s\n", nam_str.c_str());
    return nam_str + "_" + version_str;
}

std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info) {
    std::string version_id = info.version_id ? info.version_id : "(empty)";
    s << "DiscreteTypeInfo{name: " << info.name << ", version_id: " << version_id << ", parent: ";
    if (!info.parent)
        s << info.parent;
    else
        s << *info.parent;

    s << "}";
    return s;
}

// parent is commented to fix type relaxed operations
bool DiscreteTypeInfo::operator<(const DiscreteTypeInfo& b) const {
    if (name != nullptr && b.name != nullptr) {
        int cmp_status = strcmp(name, b.name);
        if (cmp_status < 0)
            return true;
        if (cmp_status == 0) {
            std::string v_id(version_id == nullptr ? "" : version_id);
            std::string bv_id(b.version_id == nullptr ? "" : b.version_id);
            if (v_id < bv_id)
                return true;
        }
    }

    return false;
}
bool DiscreteTypeInfo::operator==(const DiscreteTypeInfo& b) const {
    if (hash_value != 0 && b.hash_value != 0)
        return hash() == b.hash();
    if (name != nullptr && b.name != nullptr) {
        if (strcmp(name, b.name) == 0) {
            std::string v_id(version_id == nullptr ? "" : version_id);
            std::string bv_id(b.version_id == nullptr ? "" : b.version_id);
            if (v_id == bv_id)
                return true;
        }
    }
    return false;
}
bool DiscreteTypeInfo::operator<=(const DiscreteTypeInfo& b) const {
    return *this == b || *this < b;
}
bool DiscreteTypeInfo::operator>(const DiscreteTypeInfo& b) const {
    return !(*this <= b);
}
bool DiscreteTypeInfo::operator>=(const DiscreteTypeInfo& b) const {
    return !(*this < b);
}
bool DiscreteTypeInfo::operator!=(const DiscreteTypeInfo& b) const {
    return !(*this == b);
}
}  // namespace ov
