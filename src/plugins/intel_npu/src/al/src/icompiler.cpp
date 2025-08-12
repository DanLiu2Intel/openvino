// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/icompiler.hpp"
#include "intel_npu/simple_network_metadata_serializer.hpp"

namespace intel_npu {

void NetworkMetadata::bindRelatedDescriptors() {
    size_t ioIndex = 0;

    for (IODescriptor& input : inputs) {
        if (input.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (input.isStateInput) {
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& output) {
                    return output.isStateOutput && (output.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                input.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        } else if (input.isShapeTensor) {
            const auto relatedDescriptorIterator =
                std::find_if(inputs.begin(), inputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != inputs.end()) {
                input.relatedDescriptorIndex = std::distance(inputs.begin(), relatedDescriptorIterator);
                inputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (IODescriptor& output : outputs) {
        if (output.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (output.isShapeTensor) {
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == output.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                output.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*output.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }
}

void NetworkMetadata::saveToJson(const std::string& filePath) const {
    SimpleNetworkMetadataSerializer::saveToJson(*this, filePath);
}

void NetworkMetadata::saveToBinary(const std::string& filePath) const {
    SimpleNetworkMetadataSerializer::saveToBinary(*this, filePath);
}

NetworkMetadata NetworkMetadata::loadFromJson(const std::string& filePath) {
    // Note: For full JSON deserialization, use NetworkMetadataSerializer::loadFromJson
    // This is a placeholder implementation
    throw std::runtime_error("JSON deserialization not implemented in simple serializer. Use NetworkMetadataSerializer::loadFromJson instead.");
}

NetworkMetadata NetworkMetadata::loadFromBinary(const std::string& filePath) {
    // Note: For full binary deserialization, use NetworkMetadataSerializer::loadFromBinary
    // This is a placeholder implementation  
    throw std::runtime_error("Binary deserialization not implemented in simple serializer. Use NetworkMetadataSerializer::loadFromBinary instead.");
}

}  // namespace intel_npu
