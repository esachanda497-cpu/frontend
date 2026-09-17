const DEFAULT_LOCAL_BACKEND = "http://127.0.0.1:8000";
const CUSTOM_BACKEND_URL =
    (typeof window !== "undefined" && window.__SAFESTEP_BACKEND_URL__)
        ? String(window.__SAFESTEP_BACKEND_URL__).trim()
        : "";

const BACKEND_BASE_URL =
    CUSTOM_BACKEND_URL || DEFAULT_LOCAL_BACKEND;

const DETECTION_URL =
    `ws://127.0.0.1:8000/ws/detect`;

const startDetection =
    document.getElementById(
        "startDetection"
    );

const stopDetection =
    document.getElementById(    
        "stopDetection"
    );

const heroStartButton =
    document.getElementById(
        "heroStartButton"
    );

const detectionVideo =
    document.getElementById(
        "detectionVideo"
    );

const detectionCanvas =
    document.getElementById(
        "detectionCanvas"
    );

const detectionPlaceholder =
    document.getElementById(
        "detectionPlaceholder"
    );

const detectionStatus =
    document.getElementById(
        "detectionStatus"
    );


const canvasContext =
    detectionCanvas.getContext(
        "2d"
    );


let cameraStream = null;

let detectionSocket = null;

let frameInterval = null;

let temporaryCanvas = null;

let temporaryContext = null;

let isCameraOn = false;

let isDetecting = false;

let palmDetected = false;

let handDetector = null;

let handCheckInterval = null;


let speechQueue = [];

let isSpeaking = false;

let lastAlertKey = "";

let lastAlertTime = 0;

let currentSpeechPriority = 0;


const ALERT_COOLDOWN = 3500;


const RISK_PRIORITY = {

    SAFE: 0,

    CAUTION: 1,

    WARNING: 2,

    CRITICAL: 3
};


console.log(
    "SafeStep AI frontend loaded."
);


function setStatus(
    text
) {

    if (!detectionStatus) {
        return;
    }

    detectionStatus.textContent =
        text;
}


function showPlaceholder(
    show
) {

    if (!detectionPlaceholder) {
        return;
    }

    if (show) {

        detectionPlaceholder.style.display =
            "flex";

    } else {

        detectionPlaceholder.style.display =
            "none";
    }
}


function clearCanvas() {

    if (!detectionCanvas) {
        return;
    }

    canvasContext.clearRect(
        0,
        0,
        detectionCanvas.width,
        detectionCanvas.height
    );
}


function resizeCanvas() {

    if (!detectionVideo.videoWidth) {
        return;
    }

    if (!detectionVideo.videoHeight) {
        return;
    }

    detectionCanvas.width =
        detectionVideo.videoWidth;

    detectionCanvas.height =
        detectionVideo.videoHeight;
}


async function startCamera() {

    if (isCameraOn) {
        return;
    }

    try {

        setStatus(
            "Starting Camera..."
        );

        showPlaceholder(
            false
        );


        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {

            throw new Error(
                "Camera API is not available."
            );
        }


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    },

                    facingMode:
                        "user"
                },

                audio: false
            });


        detectionVideo.srcObject =
            cameraStream;


        await detectionVideo.play();


        resizeCanvas();


        isCameraOn = true;

        isDetecting = false;

        palmDetected = false;


        setStatus(
            "Show your palm ✋"
        );


        await initializeHandDetection();


        startPalmChecking();


        console.log(
            "Camera started. Waiting for palm."
        );

    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        showPlaceholder(
            true
        );


        setStatus(
            "Camera Error"
        );


        let message =
            "Unable to access the camera.";


        if (
            error.name ===
            "NotAllowedError"
        ) {

            message =
                "Camera permission was denied. Allow camera access and try again.";

        } else if (
            error.name ===
            "NotFoundError"
        ) {

            message =
                "No camera was found.";

        } else if (
            error.name ===
            "NotReadableError"
        ) {

            message =
                "The camera is already being used by another application.";

        } else if (
            error.name ===
            "SecurityError"
        ) {

            message =
                "The browser blocked camera access. Use Live Server.";
        }


        alert(
            message
        );
    }
}


async function initializeHandDetection() {

    if (
        typeof Hands ===
        "undefined"
    ) {

        throw new Error(
            "MediaPipe Hands failed to load."
        );
    }


    handDetector =
        new Hands({

            locateFile:
                function (
                    file
                ) {

                    return (
                        "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" +
                        file
                    );
                }
        });


    handDetector.setOptions({

        maxNumHands: 1,

        modelComplexity: 1,

        minDetectionConfidence: 0.65,

        minTrackingConfidence: 0.65
    });


    handDetector.onResults(
        handleHandResults
    );
}


function startPalmChecking() {

    if (handCheckInterval) {

        clearInterval(
            handCheckInterval
        );
    }


    handCheckInterval =
        setInterval(
            checkForPalm,
            150
        );
}


async function checkForPalm() {

    if (!isCameraOn) {
        return;
    }


    if (isDetecting) {
        return;
    }


    if (!handDetector) {
        return;
    }


    if (!detectionVideo.videoWidth) {
        return;
    }


    if (!detectionVideo.videoHeight) {
        return;
    }


    try {

        await handDetector.send({

            image:
                detectionVideo
        });

    } catch (error) {

        console.error(
            "Hand detection error:",
            error
        );
    }
}


function handleHandResults(
    results
) {

    if (!results) {
        return;
    }


    if (
        !results.multiHandLandmarks ||
        !results.multiHandLandmarks.length
    ) {

        palmDetected = false;

        if (!isDetecting) {

            setStatus(
                "Show your palm ✋"
            );
        }

        return;
    }


    const landmarks =
        results.multiHandLandmarks[0];


    const openPalm =
        isOpenPalm(
            landmarks
        );


    if (!openPalm) {

        palmDetected = false;

        if (!isDetecting) {

            setStatus(
                "Show an open palm ✋"
            );
        }

        return;
    }


    if (
        !palmDetected
    ) {

        palmDetected = true;

        setStatus(
            "Palm detected — Starting AI..."
        );


        startYOLODetection();
    }
}


function isOpenPalm(
    landmarks
) {

    const wrist =
        landmarks[0];


    const thumbTip =
        landmarks[4];

    const thumbMcp =
        landmarks[2];


    const indexTip =
        landmarks[8];

    const indexPip =
        landmarks[6];


    const middleTip =
        landmarks[12];

    const middlePip =
        landmarks[10];


    const ringTip =
        landmarks[16];

    const ringPip =
        landmarks[14];


    const pinkyTip =
        landmarks[20];

    const pinkyPip =
        landmarks[18];


    const indexOpen =
        distance2D(
            indexTip,
            wrist
        ) >
        distance2D(
            indexPip,
            wrist
        );


    const middleOpen =
        distance2D(
            middleTip,
            wrist
        ) >
        distance2D(
            middlePip,
            wrist
        );


    const ringOpen =
        distance2D(
            ringTip,
            wrist
        ) >
        distance2D(
            ringPip,
            wrist
        );


    const pinkyOpen =
        distance2D(
            pinkyTip,
            wrist
        ) >
        distance2D(
            pinkyPip,
            wrist
        );


    const thumbOpen =
        distance2D(
            thumbTip,
            wrist
        ) >
        distance2D(
            thumbMcp,
            wrist
        );


    const fingersOpen =
        Number(indexOpen) +
        Number(middleOpen) +
        Number(ringOpen) +
        Number(pinkyOpen);


    return (
        fingersOpen >= 4 &&
        thumbOpen
    );
}


function distance2D(
    pointA,
    pointB
) {

    const dx =
        pointA.x -
        pointB.x;


    const dy =
        pointA.y -
        pointB.y;


    return Math.sqrt(
        dx * dx +
        dy * dy
    );
}


function startYOLODetection() {

    if (isDetecting) {
        return;
    }


    isDetecting = true;


    if (handCheckInterval) {

        clearInterval(
            handCheckInterval
        );

        handCheckInterval =
            null;
    }


    setStatus(
        "Connecting to AI..."
    );


    startDetectionSocket();
}


function startDetectionSocket() {

    if (
        detectionSocket &&
        (
            detectionSocket.readyState ===
            WebSocket.OPEN ||
            detectionSocket.readyState ===
            WebSocket.CONNECTING
        )
    ) {

        return;
    }


    detectionSocket =
        new WebSocket(
            DETECTION_URL
        );


    detectionSocket.onopen =
        function () {

            console.log(
                "YOLO WebSocket connected."
            );


            setStatus(
                "AI Detection Active"
            );


            createTemporaryCanvas();


            startFrameTransmission();
        };


    detectionSocket.onmessage =
        function (
            event
        ) {

            try {

                const result =
                    JSON.parse(
                        event.data
                    );


                drawDetections(
                    result
                );


                processVoiceAlerts(
                    result
                );

            } catch (error) {

                console.error(
                    "Result processing error:",
                    error
                );
            }
        };


    detectionSocket.onerror =
        function (
            error
        ) {

            console.error(
                "WebSocket error:",
                error
            );


            setStatus(
                "AI Connection Error"
            );
        };


    detectionSocket.onclose =
        function () {

            console.log(
                "YOLO WebSocket disconnected."
            );


            if (isDetecting) {

                setStatus(
                    "AI Disconnected"
                );
            }
        };
}


function createTemporaryCanvas() {

    temporaryCanvas =
        document.createElement(
            "canvas"
        );


    temporaryCanvas.width =
        640;


    temporaryCanvas.height =
        480;


    temporaryContext =
        temporaryCanvas.getContext(
            "2d"
        );
}


function startFrameTransmission() {

    if (frameInterval) {

        clearInterval(
            frameInterval
        );
    }


    frameInterval =
        setInterval(
            sendFrame,
            250
        );


    console.log(
        "Frame transmission started."
    );
}


function sendFrame() {

    if (!isDetecting) {
        return;
    }


    if (!cameraStream) {
        return;
    }


    if (!detectionVideo.videoWidth) {
        return;
    }


    if (!detectionVideo.videoHeight) {
        return;
    }


    if (
        !detectionSocket ||
        detectionSocket.readyState !==
        WebSocket.OPEN
    ) {

        return;
    }


    const sourceWidth =
        detectionVideo.videoWidth;


    const sourceHeight =
        detectionVideo.videoHeight;


    const targetWidth =
        640;


    const targetHeight =
        Math.round(
            sourceHeight *
            (
                targetWidth /
                sourceWidth
            )
        );


    temporaryCanvas.width =
        targetWidth;


    temporaryCanvas.height =
        targetHeight;


    temporaryContext.drawImage(

        detectionVideo,

        0,

        0,

        targetWidth,

        targetHeight
    );


    const imageData =
        temporaryCanvas.toDataURL(
            "image/jpeg",
            0.65
        );


    const base64 =
        imageData.split(
            ","
        )[1];


    if (!base64) {
        return;
    }


    try {

        detectionSocket.send(
            base64
        );

    } catch (error) {

        console.error(
            "Frame send error:",
            error
        );
    }
}


function drawDetections(
    result
) {

    if (!result) {
        return;
    }


    if (!result.detections) {

        clearCanvas();

        return;
    }


    resizeCanvas();


    clearCanvas();


    const detections =
        result.detections;


    for (
        const detection
        of detections
    ) {

        drawDetection(
            detection
        );
    }


    updateDetectionResults(
        detections
    );
}


function drawDetection(
    detection
) {

    if (!detection.box) {
        return;
    }


    const box =
        detection.box;


    let x1 =
        Number(
            box[0]
        );


    let y1 =
        Number(
            box[1]
        );


    let x2 =
        Number(
            box[2]
        );


    let y2 =
        Number(
            box[3]
        );


    if (
        !Number.isFinite(x1) ||
        !Number.isFinite(y1) ||
        !Number.isFinite(x2) ||
        !Number.isFinite(y2)
    ) {

        return;
    }


    const backendWidth =
        640;


    const backendHeight =
        detection.frame_height ||
        detection.image_height ||
        480;


    const scaleX =
        detectionCanvas.width /
        backendWidth;


    const scaleY =
        detectionCanvas.height /
        backendHeight;


    x1 *= scaleX;

    x2 *= scaleX;

    y1 *= scaleY;

    y2 *= scaleY;


    const width =
        x2 - x1;


    const height =
        y2 - y1;


    const riskLevel =
        detection.risk_level ||
        "SAFE";


    const riskColor =
        getRiskColor(
            riskLevel
        );


    canvasContext.lineWidth =
        3;


    canvasContext.strokeStyle =
        riskColor;


    canvasContext.strokeRect(
        x1,
        y1,
        width,
        height
    );


    const objectName =
        detection.object ||
        "object";


    const direction =
        getDirection(
            detection
        );


    const movement =
        detection.movement ||
        detection.approach ||
        "STATIONARY";


    let label =
        objectName;


    label +=
        ` | ${direction}`;


    if (
        movement &&
        movement !==
        "STATIONARY"
    ) {

        label +=
            ` | ${movement}`;
    }


    if (
        detection.distance_meters !==
        undefined &&
        detection.distance_meters !==
        null
    ) {

        const distance =
            Number(
                detection.distance_meters
            );


        if (
            Number.isFinite(
                distance
            )
        ) {

            label +=
                ` | ${distance.toFixed(1)}m`;
        }
    }


    label +=
        ` | ${riskLevel}`;


    canvasContext.font =
        "16px Arial";


    const textWidth =
        canvasContext.measureText(
            label
        ).width;


    const textHeight =
        24;


    const labelY =
        Math.max(
            y1,
            textHeight
        );


    canvasContext.fillStyle =
        riskColor;


    canvasContext.fillRect(

        x1,

        labelY -
        textHeight,

        textWidth +
        12,

        textHeight
    );


    canvasContext.fillStyle =
        "#ffffff";


    canvasContext.fillText(

        label,

        x1 + 6,

        labelY - 6
    );
}


function getDirection(
    detection
) {

    if (!detection.box) {

        return (
            detection.direction ||
            "CENTER"
        );
    }


    const x1 =
        Number(
            detection.box[0]
        );


    const x2 =
        Number(
            detection.box[2]
        );


    if (
        !Number.isFinite(x1) ||
        !Number.isFinite(x2)
    ) {

        return (
            detection.direction ||
            "CENTER"
        );
    }


    const centerX =
        (
            x1 +
            x2
        ) / 2;


    const frameWidth =
        640;


    const normalizedX =
        centerX /
        frameWidth;


    if (
        normalizedX <
        0.40
    ) {

        return "LEFT";
    }


    if (
        normalizedX >
        0.60
    ) {

        return "RIGHT";
    }


    return "CENTER";
}


function getRiskColor(
    riskLevel
) {

    if (
        riskLevel ===
        "CRITICAL"
    ) {

        return "#ff1744";
    }


    if (
        riskLevel ===
        "WARNING"
    ) {

        return "#ff9800";
    }


    if (
        riskLevel ===
        "CAUTION"
    ) {

        return "#ffd600";
    }


    return "#00e676";
}


function updateDetectionResults(
    detections
) {

    return;
}


function processVoiceAlerts(
    result
) {

    if (!result) {
        return;
    }


    if (!result.detections) {
        return;
    }


    const dangerous =
        result.detections.filter(
            detection => {

                const risk =
                    detection.risk_level ||
                    "SAFE";


                const movement =
                    detection.movement ||
                    detection.approach ||
                    "STATIONARY";


                return (

                    risk ===
                    "CRITICAL"

                    ||

                    risk ===
                    "WARNING"

                    ||

                    movement ===
                    "APPROACHING "
                );
            }
        );


    if (!dangerous.length) {
        return;
    }


    dangerous.sort(
        (
            a,
            b
        ) => {

            const riskA =
                RISK_PRIORITY[
                    a.risk_level
                ] || 0;


            const riskB =
                RISK_PRIORITY[
                    b.risk_level
                ] || 0;


            if (
                riskA !==
                riskB
            ) {

                return (
                    riskB -
                    riskA
                );
            }


            return (
                Number(
                    b.risk_score ||
                    0
                ) -
                Number(
                    a.risk_score ||
                    0
                )
            );
        }
    );


    const detection =
        dangerous[0];


    const message =
        createVoiceMessage(
            detection
        );


    if (!message) {
        return;
    }


    queueSpeech(
        message,
        detection.risk_level ||
        "CAUTION"
    );
}


function createVoiceMessage(
    detection
) {

    const objectName =
        detection.object ||
        "object";


    const movement =
        detection.movement ||
        detection.approach ||
        "STATIONARY";


    const risk =
        detection.risk_level ||
        "CAUTION";


    const direction =
        getDirection(
            detection
        );


    let location;


    if (
        direction ===
        "LEFT"
    ) {

        location =
            "from your left";

    } else if (
        direction ===
        "RIGHT"
    ) {

        location =
            "from your right";

    } else {

        location =
            "directly ahead";
    }


    let prefix;


    if (
        risk ===
        "CRITICAL"
    ) {

        prefix =
            "Danger.";

    } else if (
        risk ===
        "WARNING"
    ) {

        prefix =
            "Warning.";

    } else {

        prefix =
            "Caution.";
    }


    if (
        movement ===
        "APPROACHING"
    ) {

        return (
            `${prefix} ` +
            `A ${objectName} is ` +
            `approaching ` +
            `${location}.`
        );
    }


    if (
        detection.ttc !==
        undefined &&
        detection.ttc !==
        null
    ) {

        const ttc =
            Number(
                detection.ttc
            );


        if (
            Number.isFinite(ttc) &&
            ttc < 2
        ) {

            return (
                `${prefix} ` +
                `A ${objectName} is ` +
                `very close ` +
                `${location}.`
            );
        }
    }


    if (
        detection.path_status ===
        "IN_PATH"
    ) {

        return (
            `${prefix} ` +
            `A ${objectName} is ` +
            `in your path ` +
            `${location}.`
        );
    }


    return (
        `${prefix} ` +
        `A ${objectName} is ` +
        `${location}.`
    );
}


function queueSpeech(
    message,
    riskLevel
) {

    const priority =
        RISK_PRIORITY[
            riskLevel
        ] || 0;


    const currentTime =
        Date.now();


    const alertKey =
        message.toLowerCase();


    if (
        alertKey ===
        lastAlertKey
        &&
        currentTime -
        lastAlertTime <
        ALERT_COOLDOWN
    ) {

        return;
    }


    lastAlertKey =
        alertKey;


    lastAlertTime =
        currentTime;


    if (
        isSpeaking &&
        priority >
        currentSpeechPriority
    ) {

        speechQueue = [];


        window.speechSynthesis.cancel();


        isSpeaking =
            false;
    }


    const alreadyQueued =
        speechQueue.some(
            item =>
                item.message ===
                message
        );


    if (alreadyQueued) {
        return;
    }


    speechQueue.push({

        message:
            message,

        priority:
            priority
    });


    speechQueue.sort(
        (
            a,
            b
        ) =>
            b.priority -
            a.priority
    );


    processSpeechQueue();
}


function processSpeechQueue() {

    if (isSpeaking) {
        return;
    }


    if (!speechQueue.length) {

        currentSpeechPriority =
            0;

        return;
    }


    const item =
        speechQueue.shift();


    isSpeaking =
        true;


    currentSpeechPriority =
        item.priority;


    speakMessage(
        item.message
    );
}


function speakMessage(
    message
) {

    if (
        !window.speechSynthesis
    ) {

        isSpeaking =
            false;

        processSpeechQueue();

        return;
    }


    const utterance =
        new SpeechSynthesisUtterance(
            message
        );


    utterance.lang =
        "en-US";


    utterance.rate =
        0.95;


    utterance.pitch =
        1;


    utterance.volume =
        1;


    utterance.onstart =
        function () {

            console.log(
                "VOICE:",
                message
            );
        };


    utterance.onend =
        function () {

            isSpeaking =
                false;


            currentSpeechPriority =
                0;


            setTimeout(
                processSpeechQueue,
                100
            );
        };


    utterance.onerror =
        function (
            error
        ) {

            console.error(
                "Speech error:",
                error
            );


            isSpeaking =
                false;


            currentSpeechPriority =
                0;


            setTimeout(
                processSpeechQueue,
                100
            );
        };


    window.speechSynthesis.speak(
        utterance
    );
}


function stopCamera() {

    isCameraOn =
        false;


    isDetecting =
        false;


    palmDetected =
        false;


    if (handCheckInterval) {

        clearInterval(
            handCheckInterval
        );

        handCheckInterval =
            null;
    }


    if (frameInterval) {

        clearInterval(
            frameInterval
        );

        frameInterval =
            null;
    }


    if (detectionSocket) {

        try {

            detectionSocket.close();

        } catch (error) {

            console.error(
                error
            );
        }


        detectionSocket =
            null;
    }


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => {
                    track.stop();
                }
            );


        cameraStream =
            null;
    }


    detectionVideo.srcObject =
        null;


    clearCanvas();


    speechQueue = [];


    lastAlertKey =
        "";


    lastAlertTime =
        0;


    currentSpeechPriority =
        0;


    if (
        window.speechSynthesis
    ) {

        window.speechSynthesis.cancel();
    }


    isSpeaking =
        false;


    setStatus(
        "Camera Ready"
    );


    showPlaceholder(
        true
    );


    console.log(
        "Camera stopped."
    );
}


if (startDetection) {

    startDetection.addEventListener(
        "click",
        startCamera
    );
}


if (stopDetection) {

    stopDetection.addEventListener(
        "click",
        stopCamera
    );
}


if (heroStartButton) {

    heroStartButton.addEventListener(
        "click",
        function () {

            setTimeout(
                startCamera,
                300
            );
        }
    );
}


window.addEventListener(
    "beforeunload",
    stopCamera
);
