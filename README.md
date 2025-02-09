# Detect_the_PPE_KIT_in_Automobile-Manufacturing
## Introduction
This project aims to detect Personal Protective Equipment (PPE) kits in an automobile manufacturing environment using computer vision techniques.The project focuses on enhancing safety protocols in industrial and manufacturing environments through the use of deep learning and object detection. The system aims to ensure compliance with safety regulations by detecting unauthorized human presence and verifying the use of Personal Protective Equipment (PPE) on workers.

## Project Objectives
1. **Restricted Zone Monitoring**: Detect unauthorized human presence in defined No-Human Zones.
2. **PPE Compliance Detection**: Identify and verify the presence of required safety gear such as hardhats, gloves, safety vests, masks, and goggles.
3. **Identity Verification**: Authenticate workers using facial recognition technology to ensure compliance with safety regulations and access control policies.

## Features
### 1. Human Detection in Restricted Zones
- Monitor industrial video feeds in real-time to detect humans entering No-Human Zones.
- Define and configure restricted areas within the system for targeted monitoring.
- Utilize deep learning models trained on datasets from industrial environments for high accuracy.

### 2. PPE Kit Detection
- Detect mandatory safety equipment, including helmets, radium jackets, boots, and goggles.
- Employ object detection models to classify PPE gear accurately.
- Validate detection accuracy using labeled datasets to minimize false positives and negatives.

### 3. Facial Recognition for Worker Identity Verification
- Authenticate worker identities using advanced facial recognition algorithms.
- Ensure the system performs reliably in challenging conditions, such as varying lighting or partial occlusions.
- Integrate the module with human detection and PPE compliance monitoring for a unified safety solution.

## Benefits
- **Increased Safety**: Automates the enforcement of safety protocols, reducing the likelihood of accidents.
- **Efficiency**: Real-time detection ensures immediate alerts for non-compliance.
- **Regulatory Adherence**: Supports compliance with industry safety standards.
- **Integrated Approach**: Combines multiple detection and recognition capabilities into a seamless system.

## Technology Stack
- **Frameworks:** TensorFlow, PyTorch
- **Programming Language**: Python
- **Tools**: OpenCV, YOLO (You Only Look Once) for object detection
- **Hardware**: Industrial cameras and GPUs for real-time analysis

## Installation
Installation refers to the process of setting up software, equipment, or tools to make them operational. It involves downloading, configuring, and integrating the necessary components.
Key Steps in Software Installation:
1. Download: Obtain the installation file from a trusted source.
2. Run Installer: Launch the installation program.
3. Follow Instructions: Choose preferences like installation location and additional features.
4. Install Dependencies: Ensure required libraries or tools are installed.
5. Configuration: Set up initial settings and preferences.
6. Verification: Test to confirm proper functionality.
For hardware or equipment, installation involves physical setup, connecting components, and ensuring compatibility with existing systems.

## Results-
Detected items will be displayed with bounding boxes and labels.

## Future Scope
- Expand the detection system to include additional PPE items or safety regulations.
- Integrate with IoT devices for enhanced monitoring and automated response mechanisms.
- Leverage edge computing for improved performance in low-latency environments.
