import argparse
from imutils import resize
from imutils.video import WebcamVideoStream
import cv2
from HandControl import HandControlModule


def main():
    parser = argparse.ArgumentParser()
    # MPH PARAMETERS
    parser.add_argument("--mode", type=int, default=0, help="Model complexity: 0 or 1")
    parser.add_argument("--hands", type=int, default=1, help="Number of hands to be used")
    parser.add_argument("--detConf", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--trackConf", type=float, default=0.3, help="Tracking confidence")
    # SWIPER MOVEMENT PARAMETERS
    parser.add_argument("--window", type=int, default=5, help="Smoothing window")
    parser.add_argument("--curV", type=int, default=200, help="Cursor Velocity")
    parser.add_argument("--scrV", type=int, default=100, help="Scrolling Velocity")

    parser.add_argument("--clickD", type=float, default=0.05, help="Distance for clicking")
    parser.add_argument("--swipeD", type=float, default=0.15, help="Distance for swiping")
    parser.add_argument("--direction", type=str, default="horizontal",
                        help="Direction for swiping (vertical or horizontal)")
    parser.add_argument("--showCam", type=bool, default=False,
                        help="Turn on, if you want to practice with gestures and watch how gestures works")

    args = parser.parse_args()
    print("""
Ok, we're ready to go.
If you use showCam, to quit press Q.
If showCam is False, you have to break the process (Ctrl + C)
    """)

    hpm = HandControlModule(
        mode=args.mode,
        num_hands=args.hands,
        det_conf=args.detConf,
        track_conf=args.trackConf,
        window=args.window,
        cursor_velocity=args.curV,
        scrolling_velocity=args.scrV,
        click_distance=args.clickD,
        swipe_distance=args.swipeD,
        swipe_direction=args.direction
    )

    cap = WebcamVideoStream(src=0).start()

    while True:
        image = cap.read()
        image = resize(image, 960)
        image = hpm.find_hand(image)

        # Get all coordinates of palm points and check for state
        landmarks = hpm.get_landmarks()
        label = hpm.predict_label(landmarks)
        # HANDLE GESTURE
        hpm.handle_gesture(image, label)

        if args.showCam:
            cv2.putText(image, "Predict: " + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Show image and break if need to
            cv2.imshow("Image", image)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.stop()


if __name__ == "__main__":
    main()
