🦯 SafeStep AI
Real-Time Autonomous Hazard Alerting & Conversational Assistant for Low-Vision Pedestrians
Cognitia 2026 | NLP & Computer Vision | Team NeuraSight

SEE · UNDERSTAND · PRIORITIZE · SPEAK

SafeStep AI is an AI-powered assistive system that transforms a live view of the user's surroundings into short, prioritized spoken guidance, helping blind and low-vision pedestrians better understand what matters around them while they continue walking.

SafeStep is designed as an assistive tool, not a replacement for a cane, guide dog, or the user's own judgment.

he Problem
For blind and low-vision pedestrians, navigating an unfamiliar environment creates a continuous awareness challenge.

Sudden Obstacles
Obstacles can appear with little or no warning.

Traffic From Every Direction
Cars, trucks, motorcycles and bicycles can approach from different directions.

Proximity Is Difficult to Judge
Without visual perception, determining how close an approaching object is can be difficult.

Signs & Signals Are Visual
Traffic signs, crossing signals and environmental information are often inaccessible without visual assistance.

Critical Sounds Come From Everywhere
Horns, sirens and engines can provide important information, including from outside the camera's field of view.

Alert Overload
A system that announces everything can become almost as unhelpful as one that announces nothing.

The challenge isn't seeing more. It's knowing what matters — right now.

💡 Our Solution
SafeStep AI continuously interprets the user's environment and converts relevant visual and audio information into prioritized spoken guidance.

                 ┌─────────────────────┐
                 │     CAMERA + MIC     │
                 │   Live Environment   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      AI ENGINE      │
                 │                     │
                 │  Perception         │
                 │  Hazard Understanding│
                 │  Prioritization     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │        VOICE        │
                 │  Spoken Guidance    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      PEDESTRIAN     │
                 │ Informed & Hands-Free│
                 └─────────────────────┘
The innovation is not simply YOLO detection.

SafeStep combines:

Real-time computer vision
Object tracking
Spatial reasoning
Relative proximity estimation
Hazard prioritization
Temporal hazard memory
Text-to-speech
Text and sign reading
Conversational AI
Directional audio
Failure awareness
into one assistive interface.

🏗️ System Architecture
                         SAFESTEP AI
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      ┌──────────────┐                  ┌──────────────┐
      │    CAMERA    │                  │  MICROPHONE  │
      └──────┬───────┘                  └──────┬───────┘
             │                                 │
             ▼                                 ▼
         OpenCV                         Sound Detection
             │                                 │
             ▼                                 ▼
         YOLOv11n                      Directional Audio
             │
             ▼
      Object Tracking
             │
             ▼
      Scene Understanding
             │
             └────────────┬────────────────────┘
                          ▼
                ┌───────────────────┐
                │ MULTIMODAL SCENE  │
                │      ENGINE       │
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌──────────────┐        ┌──────────────┐
       │ Hazard Engine│        │ Scene Memory │
       └──────┬───────┘        └──────┬───────┘
              │                       │
              ▼                       ▼
       Urgency / Priority      Conversational AI
              │                       │
              ▼                       ▼
        Alert Manager          Intent + Reference
              │                 Understanding
              ▼                       │
       Text-to-Speech               │
              │                     │
              └──────────┬──────────┘
                         ▼
                       USER
The architecture contains two major paths:

Safety Path
Camera → OpenCV → YOLOv11n → Tracking → Scene Understanding → Hazard Engine → Priority Engine → Alert Manager → Text-to-Speech

Conversational Path
User Speech → Speech-to-Text → Intent & Reference Understanding → Scene Memory → Response Generation

Safety alerts are always designed to preempt conversational responses.

👁️ Real-Time Computer Vision
From Camera Frames to Spatial Awareness
SafeStep uses YOLOv11n + OpenCV + multi-object tracking to understand the surrounding environment.

The system can detect objects such as:

People
Cars
Trucks
Buses
Motorcycles
Bicycles
Each tracked object receives a persistent tracking ID.

For example:

OBJECT
Truck

TRACK ID
#07

DIRECTION
Center

RELATIVE PROXIMITY
Close

MOVEMENT
Approaching

HAZARD
High
📍 Spatial Awareness
SafeStep does not stop at identifying an object.

It additionally determines:

Direction
LEFT
CENTER
RIGHT
Relative Proximity
FAR
MEDIUM
CLOSE
VERY CLOSE
Movement
APPROACHING
STABLE
MOVING AWAY
Proximity is estimated from bounding-box size and position and is explicitly treated as relative proximity rather than metric depth.

🧠 Hazard Intelligence
Detection Isn't Enough. We Decide What Matters.
Every detected object does not deserve the same level of attention.

SafeStep combines five signals:

Object Type
     +
Position
     +
Proximity
     +
Movement
     +
Temporal History
     │
     ▼
HAZARD SCORE
The resulting hazard score is categorized as:

LOW
MEDIUM
HIGH
Example
Object	Position	Proximity	Movement	Hazard
Person	Left	Far	Stable	Low
Car	Right	Medium	Approaching	Medium
Truck	Center	Close	Approaching	High
A truck approaching in the center of the user's path therefore receives a higher priority than a distant stationary person.

🔔 Smart, Non-Repetitive Alerts
Temporal Hazard Memory
A naive frame-by-frame detector could repeatedly announce:

"Car ahead."
"Car ahead."
"Car ahead."
"Car ahead."
...
This creates alert fatigue.

SafeStep instead maintains temporal memory for tracked hazards.

OBJECT APPEARS
      ↓
TRACKED
      ↓
ALERT ISSUED
      ↓
HAZARD PERSISTS
      ↓
NO UNNECESSARY REPETITION
      ↓
HAZARD BECOMES MORE DANGEROUS
      ↓
ALERT ESCALATED
Example:

"Vehicle ahead."

Later, if the vehicle becomes significantly closer:

"Warning. Vehicle very close ahead."

This alert-memory mechanism is currently a prototype feature.

🗣️ Conversational Assistant
Talk Naturally. Keep Walking.
SafeStep is designed to eventually allow users to ask natural questions about their environment.

Example:

USER:
"What's ahead?"

SAFESTEP:
"There's a truck ahead."
USER:
"What is on my left?"

SAFESTEP:
"There's a person on your left."
USER:
"What does that sign say?"

SAFESTEP:
"The sign indicates a pedestrian crossing."
The system is designed to resolve references such as:

"that sign"
"that vehicle"
"the crossing"
"it"
using recent scene memory + conversation context.

For safety-critical situations, the assistant should avoid overclaiming. For example:

"The crossing signal appears active, but please verify your surroundings."

The conversational layer is currently in development.

📖 Environmental Text & Sign Reading
See Text. Hear the World.
Objects are only part of the environment.

SafeStep is being extended to understand:

Traffic signs
Crossing signals
Traffic lights
Environmental text
Other important scene information
Example:

"The pedestrian crossing signal is active."

Text and sign reading is currently at the prototype stage, with OCR accuracy still being improved.

🔊 Directional Audio
Vision cannot see everything.

A camera may not detect:

A siren behind the user
A horn from the right
An approaching vehicle outside the camera's field of view
SafeStep's planned audio pipeline includes:

Environmental Sound
        ↓
Sound Detection
        ↓
Direction Estimation
        ↓
Hazard Interpretation
        ↓
Spoken Alert
Example:

"Siren behind you, on your left."

Directional audio is currently in development.

⚠️ Failure Awareness
Safe Systems Must Know When They Can't See
A safety system should not silently fail or pretend to be confident when its sensors become unreliable.

SafeStep is designed to detect situations such as:

Camera Blocked
"Camera visibility is limited."

Dark Environment
"Low visibility detected."

Glare
"Visual detection may be unreliable."

Audio Unavailable
"Environmental audio unavailable."

The system also uses confidence-aware communication:

DETECTED
Clear evidence — stated plainly

LIKELY
Probable — communicated as "likely"

UNCERTAIN
Flagged as uncertain

SYSTEM UNAVAILABLE
Explicitly announced
Failure awareness is currently in development.

🎙️ Text-to-Speech
SafeStep converts important AI outputs into spoken alerts so the user can receive information hands-free.

Example:

AI DETECTION
     ↓
Truck
     ↓
Center
     ↓
Close
     ↓
Approaching
     ↓
High Hazard
     ↓
TEXT-TO-SPEECH
     ↓
"Warning. Vehicle very close ahead."
Text-to-speech is part of the currently implemented system.

🌐 Web Interface
SafeStep includes a web-based interface for demonstrating the computer-vision pipeline.

The current interface supports:

SafeStep AI landing interface
Live camera detection
Camera start/stop controls
Demo video detection
Real-time detection visualization
YOLO-powered object detection
Spoken detection alerts
The web interface and road-footage demonstration are listed as implemented in the project roadmap.

🎬 Live Road Detection Demo
SafeStep has been demonstrated using real road footage.

The live detection pipeline demonstrates:

YOLOv11n
    ↓
Object Detection
    ↓
Persistent Tracking IDs
    ↓
LEFT / CENTER / RIGHT
    ↓
Relative Proximity
    ↓
Movement Estimation
    ↓
Hazard Labels
    ↓
Text-to-Speech
The demonstrated system includes:

YOLOv11n object detection
Persistent tracking
Spatial awareness
Relative proximity
Movement estimation
Text-to-speech alerts
Formal FPS, end-to-end latency and detection-precision numbers should only be reported once measured on the actual test setup.

🧩 Technology Stack
Computer Vision
Python
YOLOv11n
OpenCV
Object Tracking
Computer Vision
AI / Intelligence
Hazard Scoring
Temporal Scene Memory
Spatial Reasoning
Proximity Estimation
Text-to-Speech
OCR / Sign Reading
Conversational AI
Web
React / Frontend Web Technologies
JavaScript
HTML5
CSS3
Planned / Development Components
Speech-to-Text
Environmental Sound Detection
Directional Audio Localization
Conversational Scene Queries
Improved OCR
Uncertainty Modeling
📂 Project Structure
SafeStep AI/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
│
├── backend/
│   ├── server.js
│   ├── package.json
│   └── ...
│
├── model/
│   ├── weights/
│   └── ...
│
├── detection/
│   ├── tracking/
│   ├── hazard/
│   └── ...
│
├── docs/
│   └── ...
│
├── .gitignore
├── README.md
└── LICENSE
The exact implementation structure may evolve as additional AI and multimodal components are integrated.

🔄 End-to-End Workflow
                  LIVE ENVIRONMENT
                         │
              ┌──────────┴──────────┐
              │                     │
           CAMERA                MICROPHONE
              │                     │
              ▼                     ▼
           OpenCV              Audio Processing
              │                     │
              ▼                     ▼
          YOLOv11n            Sound Detection
              │                     │
              ▼                     ▼
       Object Tracking       Direction Estimation
              │                     │
              └──────────┬──────────┘
                         ▼
                SCENE UNDERSTANDING
                         │
                         ▼
                   SCENE MEMORY
                         │
                         ▼
                  HAZARD ENGINE
                         │
                         ▼
                PRIORITY ENGINE
                         │
                         ▼
                  ALERT MANAGER
                         │
                         ▼
                  TEXT-TO-SPEECH
                         │
                         ▼
                       USER
🟢 Current Implementation Status
Feature	Status
YOLOv11n Detection	✅ Implemented
OpenCV Camera Pipeline	✅ Implemented
Object Tracking	✅ Implemented
Persistent Track IDs	✅ Implemented
Left / Center / Right Position	✅ Implemented
Relative Proximity	✅ Implemented
Movement Estimation	✅ Implemented
Hazard Scoring	✅ Implemented
Text-to-Speech	✅ Implemented
Web Interface	✅ Implemented
Road-Footage Demo	✅ Implemented
Temporal Alert Memory	🟦 Prototype
OCR / Sign Reading	🟦 Prototype
Conversational Scene Queries	🟣 In Development
Directional Audio	🟣 In Development
Improved Approach Estimation	🟣 In Development
Failure Awareness	🟣 In Development
Advanced Uncertainty Modeling	🟣 In Development
These implementation stages follow the project's presentation distinction between implemented, prototype and in-development functionality.

🚀 Roadmap
NOW — Implemented
YOLOv11n detection
Object tracking
Relative proximity
Hazard scoring
Text-to-speech
Web interface
Road-footage demonstration
NEXT — In Development
Improved OCR
Conversational scene queries
Directional audio localization
Better approach-vector estimation
Improved uncertainty modeling
FUTURE
Personalized hazard sensitivity
More robust outdoor deployment
Better multimodal fusion
Larger real-world evaluation dataset
🎯 Impact
SafeStep AI aims to provide:

👐 Hands-Free Environmental Awareness
Information about the surrounding environment without requiring constant interaction with a phone.

🚨 Prioritized Safety Information
The most important hazards are communicated first.

🔕 Reduced Alert Fatigue
Temporal hazard memory prevents unnecessary repetition.

🗣️ Conversational Interaction
Users can naturally ask questions about their surroundings.

📖 Accessible Environmental Text
Important signs and visual information can eventually be read aloud.

🌐 Multimodal Perception
Vision and audio can work together to create richer environmental awareness.

🧠 Why SafeStep AI?
Traditional object detection answers:

"What objects are visible?"

SafeStep aims to answer:

"What does the user need to know right now?"

That distinction is the core of the project.

OBJECT DETECTION
      ↓
"What is there?"
      ↓
SPATIAL UNDERSTANDING
      ↓
"Where is it?"
      ↓
PROXIMITY + MOVEMENT
      ↓
"Is it getting closer?"
      ↓
HAZARD INTELLIGENCE
      ↓
"How important is it?"
      ↓
PRIORITIZATION
      ↓
"What should be said?"
      ↓
SPOKEN GUIDANCE
      ↓
"What does the user need to know?"
🏆 Hackathon
SafeStep AI was developed by Team NeuraSight for Cognitia 2026, focusing on the intersection of:

Natural Language Processing
Computer Vision
Accessibility
Artificial Intelligence
Real-Time Perception
Human-AI Interaction
The project demonstrates how computer vision can move beyond simple object detection toward an assistive, context-aware interface for environmental awareness.

👥 Team NeuraSight
Member	Role
shounak	Team Lead / Project Coordination/ yolo
debjyoti	Backend / System Development
Esa	Fast API and Deployment
Debarghya	Frontend / UI & Web Development
Organization: Cognitia-IEM2 Hackathon: Cognitia 2026

⚠️ Important Disclaimer
SafeStep AI is an assistive technology prototype.

It should not be considered a replacement for:

A white cane
A guide dog
Human assistance
The user's own judgment
Professional accessibility equipment
AI perception can be uncertain or incorrect. The system is therefore designed around the principle that it should communicate uncertainty rather than pretend to be confident.

📜 License
This project was developed by Team NeuraSight for the Cognitia 2026 Hackathon.

See the LICENSE file for the applicable licensing terms.

🦯 SafeStep AI
SEE · UNDERSTAND · PRIORITIZE · SPEAK
SafeStep AI turns what the world looks like into what the user needs to know.

Built by Team NeuraSight · Cognitia 2026

About
safe-step1.vercel.app
Resources
Readme
License
Activity
Stars
0 stars
Watchers
0 watching
Forks
0 forks
Report repository
Releases
No releases published
Deployments
3
 (3)
Production
last week
Packages
No packages published
Contributors
2
 (2)
@shounakhere
shounakhereShounak Sarkar
@debarghyabose
debarghyaboseDebarghya Bose
Languages
Python
71.8%
JavaScript
14%
CSS
8.2%
HTML
6%
Footer
© 2026 GitHub, In

