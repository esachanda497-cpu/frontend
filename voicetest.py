import speech_recognition as sr

recognizer = sr.Recognizer()

microphone = sr.Microphone()

print("SafeStep Voice Assistant")
print("Say a command.")

with microphone as source:

    recognizer.adjust_for_ambient_noise(
        source,
        duration=1
    )

while True:

    print("Listening...")

    try:

        with microphone as source:

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

        text = recognizer.recognize_google(
            audio
        )

        text = text.lower()

        print(
            f"You said: {text}"
        )

        if "ahead" in text or "in front" in text:

            intent = "SCENE_QUERY"

        elif "left" in text:

            intent = "LEFT_QUERY"

        elif "right" in text:

            intent = "RIGHT_QUERY"

        elif "read" in text or "sign" in text:

            intent = "OCR_QUERY"

        elif "car" in text:

            intent = "OBJECT_QUERY"

        elif "person" in text:

            intent = "OBJECT_QUERY"

        elif "stop" in text:

            intent = "EXIT"

        else:

            intent = "UNKNOWN"

        print(
            f"Intent: {intent}"
        )

        if intent == "EXIT":

            print(
                "SafeStep voice assistant stopped."
            )

            break

    except sr.WaitTimeoutError:

        print(
            "No speech detected."
        )

    except sr.UnknownValueError:

        print(
            "I could not understand that."
        )

    except sr.RequestError:

        print(
            "Speech recognition service is unavailable."
        )
