from ultralytics import YOLO
import cv2
import threading
import queue
import time

from backend.scene_memory import SceneMemory
from backend.assistant import VoiceAssistant
from backend.tts import speak
from backend.hazard_engine import HazardEngine
from backend.alert_manager import AlertManager


model = YOLO("yolo11n.pt")


camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)


camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not camera.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    raise SystemExit


scene = SceneMemory()

hazard_engine = HazardEngine()

alert_manager = AlertManager()

alert_queue = queue.PriorityQueue()

previous_objects = {}

distance_history = {}

movement_history = {}

last_seen = {}


reference_heights = {

    "person": 1.70,

    "bicycle": 1.10,

    "motorcycle": 1.20,

    "car": 1.50,

    "bus": 3.00,

    "truck": 3.00,

    "cell phone": 0.15,

    "bottle": 0.25,

    "chair": 0.90,

    "backpack": 0.50,

    "dog": 0.60,

    "cat": 0.30
}


FOCAL_LENGTH = 700.0


def estimate_distance(
    object_name,
    pixel_height
):

    if pixel_height <= 1:

        return None

    reference_height = (
        reference_heights.get(
            object_name
        )
    )

    if reference_height is None:

        return None

    distance = (
        reference_height
        * FOCAL_LENGTH
        / pixel_height
    )

    if distance <= 0:

        return None

    if distance > 20:

        return None

    return distance


def smooth_distance(
    track_id,
    distance
):

    if distance is None:

        return None

    history = (
        distance_history.get(
            track_id
        )
    )

    if history is None:

        history = []

        distance_history[
            track_id
        ] = history

    history.append(
        distance
    )

    if len(history) > 8:

        history.pop(0)

    sorted_history = sorted(
        history
    )

    middle = len(
        sorted_history
    ) // 2

    if len(sorted_history) % 2 == 0:

        return (
            sorted_history[
                middle - 1
            ]
            +
            sorted_history[
                middle
            ]
        ) / 2

    return sorted_history[
        middle
    ]


def get_distance_level(
    distance
):

    if distance is None:

        return "UNKNOWN"

    if distance < 1.0:

        return "VERY CLOSE"

    if distance < 2.0:

        return "CLOSE"

    if distance < 4.0:

        return "MEDIUM"

    if distance < 8.0:

        return "FAR"

    return "VERY FAR"


def get_direction(
    center_x,
    frame_width
):

    normalized_x = (
        center_x
        / frame_width
    )

    if normalized_x < 0.40:

        return "LEFT"

    if normalized_x > 0.60:

        return "RIGHT"

    return "CENTER"


def get_approach(
    track_id,
    current_distance,
    current_time
):

    if current_distance is None:

        return "STATIONARY", 0.0

    history = (
        movement_history.get(
            track_id
        )
    )

    if history is None:

        history = []

        movement_history[
            track_id
        ] = history

    history.append(
        (
            current_time,
            current_distance
        )
    )

    while (
        len(history) > 0
        and current_time
        - history[0][0]
        > 2.0
    ):

        history.pop(0)

    if len(history) < 3:

        return "STATIONARY", 0.0

    old_time, old_distance = (
        history[0]
    )

    delta_time = (
        current_time
        - old_time
    )

    if delta_time <= 0:

        return "STATIONARY", 0.0

    closing_speed = (
        old_distance
        - current_distance
    ) / delta_time

    if closing_speed > 0.15:

        return (
            "APPROACHING",
            closing_speed
        )

    if closing_speed < -0.15:

        return (
            "MOVING AWAY",
            abs(closing_speed)
        )

    return "STATIONARY", 0.0


def get_x_velocity(
    track_id,
    center_x,
    current_time,
    frame_width
):

    previous = (
        previous_objects.get(
            track_id
        )
    )

    if previous is None:

        return 0.0

    previous_x = (
        previous["center_x"]
    )

    previous_time = (
        previous["time"]
    )

    delta_time = (
        current_time
        - previous_time
    )

    if delta_time <= 0:

        return 0.0

    velocity = (
        center_x
        - previous_x
    ) / delta_time

    maximum = (
        frame_width
        * 2
    )

    velocity = max(
        min(
            velocity,
            maximum
        ),
        -maximum
    )

    return velocity


def get_risk_display(
    risk_level
):

    if risk_level == "CRITICAL":

        return "CRITICAL"

    if risk_level == "WARNING":

        return "WARNING"

    if risk_level == "CAUTION":

        return "CAUTION"

    return "SAFE"


def voice_worker():

    while True:

        item = (
            alert_queue.get()
        )

        if item is None:

            alert_queue.task_done()

            break

        priority, message = item

        if message == "__STOP__":

            alert_queue.task_done()

            break

        try:

            speak(message)

        except Exception as error:

            print(
                f"\nTTS ERROR: {error}"
            )

        alert_queue.task_done()


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)

voice_thread.start()


assistant = VoiceAssistant(
    scene,
    alert_queue
)


assistant_thread = threading.Thread(
    target=assistant.run,
    daemon=True
)

assistant_thread.start()


print(
    "\nSafeStep AI started."
)

print(
    "Press Q to quit."
)

print(
    "Ask: What is in front of me?"
)

print(
    "Ask: What is on my left?"
)

print(
    "Ask: What is on my right?"
)

print(
    "Ask: Is something approaching?"
)

print(
    "Ask: Is something moving away?"
)


try:

    while True:

        success, frame = (
            camera.read()
        )

        if not success:

            print(
                "Camera frame error."
            )

            break

        frame_height, frame_width = (
            frame.shape[:2]
        )

        current_time = time.time()

        active_ids = []

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < 0.40:

                    continue

                class_id = int(
                    box.cls[0]
                )

                object_name = (
                    model.names[
                        class_id
                    ]
                )

                if box.id is None:

                    continue

                track_id = int(
                    box.id[0]
                )

                active_ids.append(
                    track_id
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(
                    0,
                    min(
                        x1,
                        frame_width - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame_width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame_height - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame_height - 1
                    )
                )

                pixel_height = (
                    y2 - y1
                )

                if pixel_height <= 1:

                    continue

                center_x = (
                    x1 + x2
                ) / 2

                bottom_y = y2

                direction = (
                    get_direction(
                        center_x,
                        frame_width
                    )
                )

                raw_distance = (
                    estimate_distance(
                        object_name,
                        pixel_height
                    )
                )

                distance = (
                    smooth_distance(
                        track_id,
                        raw_distance
                    )
                )

                distance_level = (
                    get_distance_level(
                        distance
                    )
                )

                approach, closing_speed = (
                    get_approach(
                        track_id,
                        distance,
                        current_time
                    )
                )

                ttc = (
                    hazard_engine
                    .calculate_ttc(
                        distance,
                        closing_speed
                    )
                )

                x_velocity = (
                    get_x_velocity(
                        track_id,
                        center_x,
                        current_time,
                        frame_width
                    )
                )

                path_status = (
                    hazard_engine
                    .walking_path_status(
                        center_x,
                        bottom_y,
                        frame_width,
                        frame_height
                    )
                )

                predicted_intersection = (
                    hazard_engine
                    .predict_path_intersection(
                        center_x,
                        x_velocity,
                        ttc,
                        frame_width
                    )
                )

                hazard = (
                    hazard_engine
                    .analyze(
                        object_name,
                        distance,
                        direction,
                        closing_speed,
                        ttc,
                        path_status,
                        predicted_intersection,
                        confidence
                    )
                )

                risk_score = (
                    hazard["risk_score"]
                )

                risk_level = (
                    hazard["risk_level"]
                )

                scene.update(
                    track_id,
                    object_name,
                    direction,
                    distance_level,
                    distance,
                    approach,
                    confidence,
                    risk_score,
                    risk_level,
                    ttc,
                    path_status,
                    predicted_intersection,
                    closing_speed
                )

                should_alert = (
                    alert_manager
                    .should_alert(
                        track_id,
                        risk_level,
                        risk_score,
                        object_name,
                        direction,
                        ttc
                    )
                )

                if should_alert:

                    alert_message = (
                        alert_manager
                        .create_message(
                            object_name,
                            direction,
                            distance,
                            risk_level,
                            ttc,
                            approach,
                            path_status
                        )
                    )

                    if risk_level == "CRITICAL":

                        priority = 0

                    elif risk_level == "WARNING":

                        priority = 1

                    else:

                        priority = 2

                    alert_queue.put(
                        (
                            priority,
                            alert_message
                        )
                    )

                    print(
                        f"\nALERT: "
                        f"{alert_message}"
                    )

                previous_objects[
                    track_id
                ] = {
                    "center_x":
                        center_x,
                    "center_y":
                        (
                            y1 + y2
                        ) / 2,
                    "distance":
                        distance,
                    "time":
                        current_time
                }

                last_seen[
                    track_id
                ] = current_time

                if risk_level == "CRITICAL":

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif risk_level == "WARNING":

                    box_color = (
                        0,
                        165,
                        255
                    )

                elif risk_level == "CAUTION":

                    box_color = (
                        0,
                        255,
                        255
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"{object_name} "
                        f"{confidence:.2f}"
                    ),
                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        "Distance: "
                        +
                        (
                            f"{distance:.1f}m"
                            if distance is not None
                            else "N/A"
                        )
                    ),
                    (
                        x1,
                        y2 + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Direction: "
                        f"{direction}"
                    ),
                    (
                        x1,
                        y2 + 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Movement: "
                        f"{approach}"
                    ),
                    (
                        x1,
                        y2 + 60
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Risk: "
                        f"{risk_level} "
                        f"{risk_score:.0f}"
                    ),
                    (
                        x1,
                        y2 + 80
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        "TTC: "
                        +
                        (
                            f"{ttc:.1f}s"
                            if ttc is not None
                            else "N/A"
                        )
                    ),
                    (
                        x1,
                        y2 + 100
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Path: "
                        f"{path_status}"
                    ),
                    (
                        x1,
                        y2 + 120
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )

                if predicted_intersection:

                    cv2.putText(
                        frame,
                        "TRAJECTORY -> PATH",
                        (
                            x1,
                            y2 + 140
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        box_color,
                        2
                    )

        scene.remove_missing(
            active_ids
        )

        alert_manager.cleanup(
            active_ids
        )

        current_ids = set(
            active_ids
        )

        stale_ids = []

        for track_id in list(
            previous_objects
        ):

            if track_id not in current_ids:

                last_time = (
                    last_seen.get(
                        track_id,
                        current_time
                    )
                )

                if (
                    current_time
                    - last_time
                    > 2.0
                ):

                    stale_ids.append(
                        track_id
                    )

        for track_id in stale_ids:

            previous_objects.pop(
                track_id,
                None
            )

            last_seen.pop(
                track_id,
                None
            )

            distance_history.pop(
                track_id,
                None
            )

            movement_history.pop(
                track_id,
                None
            )

        left_boundary = int(
            frame_width * 0.40
        )

        right_boundary = int(
            frame_width * 0.60
        )

        cv2.line(
            frame,
            (
                left_boundary,
                0
            ),
            (
                left_boundary,
                frame_height
            ),
            (
                255,
                255,
                255
            ),
            1
        )

        cv2.line(
            frame,
            (
                right_boundary,
                0
            ),
            (
                right_boundary,
                frame_height
            ),
            (
                255,
                255,
                255
            ),
            1
        )

        cv2.putText(
            frame,
            "LEFT",
            (
                30,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            "CENTER",
            (
                int(
                    frame_width * 0.44
                ),
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            "RIGHT",
            (
                int(
                    frame_width * 0.88
                ),
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            "SafeStep AI",
            (
                20,
                frame_height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.imshow(
            "SafeStep AI",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break


except KeyboardInterrupt:

    print(
        "\nStopping SafeStep..."
    )


finally:

    assistant.stop()

    alert_queue.put(
        (
            999,
            "__STOP__"
        )
    )

    voice_thread.join(
        timeout=3
    )

    camera.release()

    cv2.destroyAllWindows()

    print(
        "SafeStep stopped."
    )
