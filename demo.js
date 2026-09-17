
/* ========================================
   SAFESTEP AI - DEMO PAGE JAVASCRIPT
   ======================================== */


/* ========================================
   GET HTML ELEMENTS
   ======================================== */

const demoVideo =
    document.getElementById("demoVideo");

const demoPlaceholder =
    document.getElementById("demoPlaceholder");

const demoCamera =
    document.getElementById("demoCamera");

const liveCameraPlaceholder =
    document.getElementById("liveCameraPlaceholder");

const startDemoCamera =
    document.getElementById("startDemoCamera");

const stopDemoCamera =
    document.getElementById("stopDemoCamera");

const demoCameraStatus =
    document.getElementById("demoCameraStatus");


/* ========================================
   CAMERA STREAM
   ======================================== */

let demoCameraStream = null;


/* ========================================
   UPDATE CAMERA STATUS
   ======================================== */

function updateDemoStatus(message) {

    if (demoCameraStatus) {
        demoCameraStatus.textContent = message;
    }

}


/* ========================================
   SHOW CAMERA PLACEHOLDER
   ======================================== */

function showLivePlaceholder(message) {

    if (!liveCameraPlaceholder) {
        return;
    }


    liveCameraPlaceholder.classList.remove("hidden");


    const text =
        liveCameraPlaceholder.querySelector("p");


    if (text) {
        text.textContent = message;
    }

}


/* ========================================
   HIDE CAMERA PLACEHOLDER
   ======================================== */

function hideLivePlaceholder() {

    if (liveCameraPlaceholder) {
        liveCameraPlaceholder.classList.add("hidden");
    }

}


/* ========================================
   START DEMO CAMERA
   ======================================== */

async function startCamera() {

    /*
       Check browser camera support.
    */

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        updateDemoStatus("Not Supported");

        showLivePlaceholder(
            "Camera access is not supported by this browser."
        );

        return;

    }


    /*
       Don't create another camera stream
       if one is already active.
    */

    if (demoCameraStream) {

        updateDemoStatus("Active");

        hideLivePlaceholder();

        return;

    }


    updateDemoStatus("Starting...");

    showLivePlaceholder(
        "Starting camera..."
    );


    try {

        /*
           Request access to the camera.

           No microphone is requested.
        */

        demoCameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: "environment"
                },

                audio: false

            });


        /*
           Attach camera stream to video.
        */

        demoCamera.srcObject =
            demoCameraStream;


        /*
           Start displaying the camera.
        */

        try {

            await demoCamera.play();

        } catch (playError) {

            console.log(
                "Camera playback waiting:",
                playError
            );

        }


        /*
           Hide placeholder.
        */

        hideLivePlaceholder();


        /*
           Update status.
        */

        updateDemoStatus("Active");


        /*
           Update buttons.
        */

        if (startDemoCamera) {
            startDemoCamera.disabled = true;
        }


        if (stopDemoCamera) {
            stopDemoCamera.disabled = false;
        }


    } catch (error) {

        console.error(
            "Demo camera error:",
            error
        );


        /*
           Permission denied.
        */

        if (error.name === "NotAllowedError") {

            updateDemoStatus(
                "Permission Denied"
            );

            showLivePlaceholder(
                "Camera permission was denied. Please allow camera access to use this feature."
            )
