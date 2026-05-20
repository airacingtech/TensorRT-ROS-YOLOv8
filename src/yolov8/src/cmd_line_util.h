#pragma once
#include <filesystem>
#include <iostream>
#include "yolov8.h"

inline void showHelp(char* argv[]) {
    std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;

    std::cout << "Options:\n"
              << "  --model <path>              Path to the ONNX model (required)\n"
              << "  --precision <FP32|FP16|INT8> Inference precision (default: FP16)\n"
              << "  --calibration-data <path>   Calibration directory (required if precision=INT8)\n"
              << "  --prob-threshold <float>    Confidence filter (default: 0.25)\n"
              << "  --nms-threshold <float>     NMS IoU threshold (default: 0.65)\n"
              << "  --top-k <int>               Max detections to return (default: 100)\n"
              << "  --seg-channels <int>        Segmentation channels (default: 32)\n"
              << "  --seg-h <int>               Segmentation mask height (default: 160)\n"
              << "  --seg-w <int>               Segmentation mask width (default: 160)\n"
              << "  --seg-threshold <float>     Segmentation threshold (default: 0.5)\n"
              << "  --batch-size <int>          Engine batch size (default: 4)\n"
              << "  --class-names <str ...>     Class names matching the model output\n"
              << std::endl;

    std::cout << "Example:\n  " << argv[0]
              << " --model model.onnx --precision FP16 --prob-threshold 0.3 --class-names car"
              << std::endl;
}

inline bool tryGetNextArgument(int argc, char* argv[], int& currentIndex, std::string& value, std::string flag, bool printErrors = true) {
    if (currentIndex + 1 >= argc) {
        if (printErrors)
            std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
        return false;
    }

    std::string nextArgument = argv[currentIndex + 1];
    if (nextArgument.substr(0, 2) == "--") {
        if (printErrors)
            std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
        return false;
    }

    value = argv[++currentIndex];
    return true;
}

inline bool tryParseInt(const std::string& s, int& value, const std::string& flag) {
    try {
        value = std::stoi(s);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Error: Could not parse '" << s << "' as an integer for flag '" << flag << "'" << std::endl;
        return false;
    }
}

inline bool tryParseFloat(const std::string& s, float& value, const std::string& flag) {
    try {
        value = std::stof(s);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Error: Could not parse '" << s << "' as a float for flag '" << flag << "'" << std::endl;
        return false;
    }
}

inline std::string parseArguments(int argc, char* argv[], YoloV8Config& config, std::string& onnxModelPath) {
    if (argc == 1) {
        showHelp(argv);
        return "Error: No arguments provided.";
    }

    for (int i = 1; i < argc; i++) {
        std::string argument = argv[i];

        if (argument.substr(0, 2) == "--") {
            std::string flag = argument.substr(2);
            std::string nextArgument;

            if (flag == "model") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'model'.";

                if (!Util::doesFileExist(nextArgument)) {
                    return "Error: Unable to find model at path '" + nextArgument + "' for flag '" + flag + "'.";
                }

                onnxModelPath = nextArgument;
            }

            else if (flag == "prob-threshold") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'prob-threshold'.";

                float value;
                if (!tryParseFloat(nextArgument, value, flag))
                    return "Error: Unable to parse float for flag 'prob-threshold'.";

                config.probabilityThreshold = value;
            }

            else if (flag == "nms-threshold") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'nms-threshold'.";

                float value;
                if (!tryParseFloat(nextArgument, value, flag))
                    return "Error: Unable to parse float for flag 'nms-threshold'.";

                config.nmsThreshold = value;
            }

            else if (flag == "top-k") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'top-k'.";

                int value;
                if (!tryParseInt(nextArgument, value, flag))
                    return "Error: Unable to parse integer for flag 'top-k'.";

                config.topK = value;
            }

            else if (flag == "seg-channels") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'seg-channels'.";

                int value;
                if (!tryParseInt(nextArgument, value, flag))
                    return "Error: Unable to parse integer for flag 'seg-channels'.";

                config.segChannels = value;
            }

            else if (flag == "seg-h") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'seg-h'.";

                int value;
                if (!tryParseInt(nextArgument, value, flag))
                    return "Error: Unable to parse integer for flag 'seg-h'.";

                config.segH = value;
            }

            else if (flag == "precision") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'precision'.";

                if (nextArgument == "FP32") {
                    config.precision = Precision::FP32;
                } else if (nextArgument == "FP16") {
                    config.precision = Precision::FP16;
                } else if (nextArgument == "INT8") {
                    config.precision = Precision::INT8;
                } else {
                    return "Error: Unexpected precision value: " + nextArgument + ", options are FP32, FP16, INT8.";
                }
            }

            else if (flag == "calibration-data") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: No path specified for flag 'calibration-data'.";

                if (nextArgument.empty()) {
                    continue;
                }

                if (!std::filesystem::is_directory(nextArgument)) {
                    return "Error: Calibration data directory does not exist or is not a directory: " + nextArgument + ".";
                }

                config.calibrationDataDirectory = nextArgument;
            }

            else if (flag == "seg-w") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'seg-w'.";

                int value;
                if (!tryParseInt(nextArgument, value, flag))
                    return "Error: Unable to parse integer for flag 'seg-w'.";

                config.segW = value;
            }

            else if (flag == "seg-threshold") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'seg-threshold'.";

                float value;
                if (!tryParseFloat(nextArgument, value, flag))
                    return "Error: Unable to parse float for flag 'seg-threshold'.";

                config.segmentationThreshold = value;
            }

            else if (flag == "batch-size") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return "Error: Unable to get next argument for flag 'batch-size'.";

                int value;
                if (!tryParseInt(nextArgument, value, flag))
                    return "Error: Unable to parse integer for flag 'batch-size'.";

                config.batchSize = value;
            }

            else if (flag == "class-names") {
                std::vector<std::string> values;
                while (tryGetNextArgument(argc, argv, i, nextArgument, flag, false)) {
                    values.push_back(nextArgument);
                }

                if (values.size() == 0) {
                    return "Error: No arguments provided for flag 'class-names'.";
                }

                config.classNames = values;
            }

            else if (flag == "ros-args") {
                // Everything after --ros-args belongs to ROS 2 (remappings, params-file, etc.).
                // Stop parsing here so we don't trip on ROS-specific flags we don't recognize.
                break;
            }

            else {
                showHelp(argv);
                return "Error: Unknown flag '" + flag + "'.";
            }
        }
        else {
            showHelp(argv);
            return "Error: Unknown argument '" + argument + "'.";
        }
    }

    if (onnxModelPath.empty()) {
        return "Error: No arguments provided for flag 'model'.";
    }

    // Return success represented by an empty string
    return "";
}
